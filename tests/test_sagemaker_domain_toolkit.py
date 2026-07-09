from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.sagemaker_domain_toolkit import (
    PLANNED_IDS,
    AwsClients,
    DriftError,
    _wait_for_domain,
    apply_configuration,
    load_domain_config,
    main_apply,
    main_status,
)


class FakeAwsError(Exception):
    def __init__(self, code: str, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.response = {
            "Error": {"Code": code, "Message": message},
            "ResponseMetadata": {"HTTPStatusCode": status_code},
        }


class FakeMeta:
    def __init__(self, region_name: str) -> None:
        self.region_name = region_name


def _tags_to_dict(tags: list[dict[str, str]]) -> dict[str, str]:
    return {item["Key"]: item["Value"] for item in tags}


def _resource_matches_filters(resource: dict[str, Any], filters: list[dict[str, Any]]) -> bool:
    tags = _tags_to_dict(resource.get("Tags", []))
    for item in filters:
        name = item["Name"]
        values = item["Values"]
        if name.startswith("tag:"):
            if tags.get(name.split(":", 1)[1]) not in values:
                return False
            continue
        if name == "vpc-id":
            if resource.get("VpcId") not in values:
                return False
            continue
        if name == "attachment.vpc-id":
            attachments = resource.get("Attachments", [])
            if not any(attachment.get("VpcId") in values for attachment in attachments):
                return False
            continue
        if name == "service-name":
            if resource.get("ServiceName") not in values:
                return False
            continue
        if name == "state":
            if resource.get("State") not in values:
                return False
            continue
        raise AssertionError(f"Unsupported filter in fake EC2 client: {name}")
    return True


class FakeS3Client:
    def __init__(self, region_name: str) -> None:
        self.meta = FakeMeta(region_name)
        self.buckets: dict[str, dict[str, Any]] = {}
        self.inaccessible_buckets: set[str] = set()

    def head_bucket(self, *, Bucket: str) -> None:
        if Bucket in self.inaccessible_buckets:
            raise FakeAwsError("403", "Access denied", status_code=403)
        if Bucket not in self.buckets:
            raise FakeAwsError("NoSuchBucket", "Missing bucket", status_code=404)

    def create_bucket(self, *, Bucket: str, CreateBucketConfiguration: dict[str, Any] | None = None) -> None:
        location = self.meta.region_name
        if CreateBucketConfiguration is not None:
            location = CreateBucketConfiguration["LocationConstraint"]
        self.buckets[Bucket] = {
            "Bucket": Bucket,
            "LocationConstraint": None if location == "us-east-1" else location,
            "PublicAccessBlockConfiguration": None,
            "ServerSideEncryptionConfiguration": None,
            "TagSet": [],
            "VersioningStatus": None,
        }

    def get_bucket_versioning(self, *, Bucket: str) -> dict[str, Any]:
        return {"Status": self.buckets[Bucket]["VersioningStatus"]}

    def put_bucket_versioning(self, *, Bucket: str, VersioningConfiguration: dict[str, Any]) -> None:
        self.buckets[Bucket]["VersioningStatus"] = VersioningConfiguration["Status"]

    def get_bucket_encryption(self, *, Bucket: str) -> dict[str, Any]:
        config = self.buckets[Bucket]["ServerSideEncryptionConfiguration"]
        if config is None:
            raise FakeAwsError(
                "ServerSideEncryptionConfigurationNotFoundError",
                "Missing encryption",
            )
        return {"ServerSideEncryptionConfiguration": config}

    def put_bucket_encryption(self, *, Bucket: str, ServerSideEncryptionConfiguration: dict[str, Any]) -> None:
        self.buckets[Bucket]["ServerSideEncryptionConfiguration"] = ServerSideEncryptionConfiguration

    def get_public_access_block(self, *, Bucket: str) -> dict[str, Any]:
        config = self.buckets[Bucket]["PublicAccessBlockConfiguration"]
        if config is None:
            raise FakeAwsError("NoSuchPublicAccessBlockConfiguration", "Missing block")
        return {"PublicAccessBlockConfiguration": config}

    def put_public_access_block(self, *, Bucket: str, PublicAccessBlockConfiguration: dict[str, Any]) -> None:
        self.buckets[Bucket]["PublicAccessBlockConfiguration"] = PublicAccessBlockConfiguration

    def get_bucket_tagging(self, *, Bucket: str) -> dict[str, Any]:
        tag_set = self.buckets[Bucket]["TagSet"]
        if not tag_set:
            raise FakeAwsError("NoSuchTagSet", "Missing tags")
        return {"TagSet": tag_set}

    def put_bucket_tagging(self, *, Bucket: str, Tagging: dict[str, Any]) -> None:
        self.buckets[Bucket]["TagSet"] = Tagging["TagSet"]

    def get_bucket_location(self, *, Bucket: str) -> dict[str, Any]:
        return {"LocationConstraint": self.buckets[Bucket]["LocationConstraint"]}


class FakeIamClient:
    def __init__(self, region_name: str) -> None:
        self.meta = FakeMeta(region_name)
        self.roles: dict[str, dict[str, Any]] = {}

    def get_role(self, *, RoleName: str) -> dict[str, Any]:
        role = self.roles.get(RoleName)
        if role is None:
            raise FakeAwsError("NoSuchEntity", "Missing role", status_code=404)
        return {"Role": role}

    def create_role(
        self,
        *,
        RoleName: str,
        AssumeRolePolicyDocument: str,
        Description: str,
        Tags: list[dict[str, str]],
    ) -> dict[str, Any]:
        role = {
            "Arn": f"arn:aws:iam::123456789012:role/{RoleName}",
            "AssumeRolePolicyDocument": json.loads(AssumeRolePolicyDocument),
            "AttachedPolicies": set(),
            "Description": Description,
            "InlinePolicies": {},
            "RoleName": RoleName,
            "Tags": Tags,
        }
        self.roles[RoleName] = role
        return {"Role": role}

    def tag_role(self, *, RoleName: str, Tags: list[dict[str, str]]) -> None:
        self.roles[RoleName]["Tags"] = Tags

    def update_assume_role_policy(self, *, RoleName: str, PolicyDocument: str) -> None:
        self.roles[RoleName]["AssumeRolePolicyDocument"] = json.loads(PolicyDocument)

    def list_attached_role_policies(self, *, RoleName: str) -> dict[str, Any]:
        policies = self.roles[RoleName]["AttachedPolicies"]
        return {"AttachedPolicies": [{"PolicyArn": item} for item in sorted(policies)]}

    def attach_role_policy(self, *, RoleName: str, PolicyArn: str) -> None:
        self.roles[RoleName]["AttachedPolicies"].add(PolicyArn)

    def get_role_policy(self, *, RoleName: str, PolicyName: str) -> dict[str, Any]:
        inline_policies = self.roles[RoleName]["InlinePolicies"]
        if PolicyName not in inline_policies:
            raise FakeAwsError("NoSuchEntity", "Missing inline policy", status_code=404)
        return {"PolicyDocument": inline_policies[PolicyName]}

    def put_role_policy(self, *, RoleName: str, PolicyName: str, PolicyDocument: str) -> None:
        self.roles[RoleName]["InlinePolicies"][PolicyName] = json.loads(PolicyDocument)


class FakeEc2Client:
    def __init__(self, region_name: str) -> None:
        self.meta = FakeMeta(region_name)
        self.vpcs: dict[str, dict[str, Any]] = {}
        self.subnets: dict[str, dict[str, Any]] = {}
        self.internet_gateways: dict[str, dict[str, Any]] = {}
        self.route_tables: dict[str, dict[str, Any]] = {}
        self.addresses: dict[str, dict[str, Any]] = {}
        self.nat_gateways: dict[str, dict[str, Any]] = {}
        self.vpc_endpoints: dict[str, dict[str, Any]] = {}
        self.security_groups: dict[str, dict[str, Any]] = {}
        self._next_ids = {
            "assoc": 1,
            "eip": 1,
            "igw": 1,
            "nat": 1,
            "rtb": 1,
            "sg": 1,
            "subnet": 1,
            "vpce": 1,
            "vpc": 1,
        }

    def _new_id(self, prefix: str) -> str:
        next_value = self._next_ids[prefix]
        self._next_ids[prefix] += 1
        return f"{prefix}-{next_value:04d}"

    def describe_availability_zones(self, *, Filters: list[dict[str, Any]]) -> dict[str, Any]:
        assert Filters == [{"Name": "state", "Values": ["available"]}]
        return {
            "AvailabilityZones": [
                {"ZoneName": f"{self.meta.region_name}a"},
                {"ZoneName": f"{self.meta.region_name}b"},
                {"ZoneName": f"{self.meta.region_name}c"},
            ]
        }

    def describe_vpcs(self, *, Filters: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "Vpcs": [item for item in self.vpcs.values() if _resource_matches_filters(item, Filters)]
        }

    def create_vpc(self, *, CidrBlock: str, TagSpecifications: list[dict[str, Any]]) -> dict[str, Any]:
        vpc_id = self._new_id("vpc")
        vpc = {"VpcId": vpc_id, "CidrBlock": CidrBlock, "Tags": TagSpecifications[0]["Tags"]}
        self.vpcs[vpc_id] = vpc
        return {"Vpc": vpc}

    def modify_vpc_attribute(self, *, VpcId: str, **_kwargs: Any) -> None:
        assert VpcId in self.vpcs

    def describe_subnets(self, *, Filters: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "Subnets": [item for item in self.subnets.values() if _resource_matches_filters(item, Filters)]
        }

    def create_subnet(
        self,
        *,
        VpcId: str,
        CidrBlock: str,
        AvailabilityZone: str,
        TagSpecifications: list[dict[str, Any]],
    ) -> dict[str, Any]:
        subnet_id = self._new_id("subnet")
        subnet = {
            "AvailabilityZone": AvailabilityZone,
            "CidrBlock": CidrBlock,
            "MapPublicIpOnLaunch": False,
            "SubnetId": subnet_id,
            "Tags": TagSpecifications[0]["Tags"],
            "VpcId": VpcId,
        }
        self.subnets[subnet_id] = subnet
        return {"Subnet": subnet}

    def modify_subnet_attribute(self, *, SubnetId: str, MapPublicIpOnLaunch: dict[str, bool]) -> None:
        self.subnets[SubnetId]["MapPublicIpOnLaunch"] = MapPublicIpOnLaunch["Value"]

    def describe_internet_gateways(self, *, Filters: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "InternetGateways": [
                item for item in self.internet_gateways.values() if _resource_matches_filters(item, Filters)
            ]
        }

    def create_internet_gateway(self, *, TagSpecifications: list[dict[str, Any]]) -> dict[str, Any]:
        igw_id = self._new_id("igw")
        gateway = {"InternetGatewayId": igw_id, "Attachments": [], "Tags": TagSpecifications[0]["Tags"]}
        self.internet_gateways[igw_id] = gateway
        return {"InternetGateway": gateway}

    def attach_internet_gateway(self, *, InternetGatewayId: str, VpcId: str) -> None:
        self.internet_gateways[InternetGatewayId]["Attachments"] = [{"VpcId": VpcId}]

    def describe_route_tables(self, *, Filters: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "RouteTables": [item for item in self.route_tables.values() if _resource_matches_filters(item, Filters)]
        }

    def create_route_table(self, *, VpcId: str, TagSpecifications: list[dict[str, Any]]) -> dict[str, Any]:
        route_table_id = self._new_id("rtb")
        route_table = {
            "Associations": [],
            "RouteTableId": route_table_id,
            "Routes": [],
            "Tags": TagSpecifications[0]["Tags"],
            "VpcId": VpcId,
        }
        self.route_tables[route_table_id] = route_table
        return {"RouteTable": route_table}

    def create_route(self, *, RouteTableId: str, DestinationCidrBlock: str, **kwargs: Any) -> None:
        route = {"DestinationCidrBlock": DestinationCidrBlock, **kwargs}
        self.route_tables[RouteTableId]["Routes"] = [
            existing
            for existing in self.route_tables[RouteTableId]["Routes"]
            if existing["DestinationCidrBlock"] != DestinationCidrBlock
        ]
        self.route_tables[RouteTableId]["Routes"].append(route)

    def replace_route(self, *, RouteTableId: str, DestinationCidrBlock: str, **kwargs: Any) -> None:
        self.create_route(RouteTableId=RouteTableId, DestinationCidrBlock=DestinationCidrBlock, **kwargs)

    def associate_route_table(self, *, RouteTableId: str, SubnetId: str) -> None:
        association_id = self._new_id("assoc")
        association = {"RouteTableAssociationId": association_id, "SubnetId": SubnetId}
        self.route_tables[RouteTableId]["Associations"].append(association)

    def replace_route_table_association(self, *, AssociationId: str, RouteTableId: str) -> None:
        for route_table in self.route_tables.values():
            for association in route_table["Associations"]:
                if association.get("RouteTableAssociationId") == AssociationId:
                    route_table["Associations"].remove(association)
                    self.route_tables[RouteTableId]["Associations"].append(association)
                    return
        raise AssertionError(f"Unknown route table association {AssociationId}")

    def describe_addresses(self, *, Filters: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "Addresses": [item for item in self.addresses.values() if _resource_matches_filters(item, Filters)]
        }

    def allocate_address(self, *, Domain: str) -> dict[str, Any]:
        assert Domain == "vpc"
        allocation_id = self._new_id("eip")
        address = {"AllocationId": allocation_id, "Tags": []}
        self.addresses[allocation_id] = address
        return address

    def create_tags(self, *, Resources: list[str], Tags: list[dict[str, str]]) -> None:
        resources = [self.vpcs, self.subnets, self.internet_gateways, self.route_tables, self.addresses]
        for resource_id in Resources:
            for collection in resources:
                if resource_id in collection:
                    collection[resource_id]["Tags"] = Tags
                    break
            else:
                raise AssertionError(f"Unknown resource for tagging: {resource_id}")

    def describe_nat_gateways(
        self,
        *,
        Filter: list[dict[str, Any]] | None = None,
        Filters: list[dict[str, Any]] | None = None,
        NatGatewayIds: list[str] | None = None,
    ) -> dict[str, Any]:
        items = list(self.nat_gateways.values())
        if NatGatewayIds is not None:
            items = [item for item in items if item["NatGatewayId"] in NatGatewayIds]
        active_filters = Filter or Filters or []
        if active_filters:
            items = [item for item in items if _resource_matches_filters(item, active_filters)]
        return {"NatGateways": items}

    def create_nat_gateway(
        self,
        *,
        SubnetId: str,
        AllocationId: str,
        TagSpecifications: list[dict[str, Any]],
    ) -> dict[str, Any]:
        nat_gateway_id = self._new_id("nat")
        nat_gateway = {
            "NatGatewayAddresses": [{"AllocationId": AllocationId}],
            "NatGatewayId": nat_gateway_id,
            "State": "available",
            "SubnetId": SubnetId,
            "Tags": TagSpecifications[0]["Tags"],
            "VpcId": self.subnets[SubnetId]["VpcId"],
        }
        self.nat_gateways[nat_gateway_id] = nat_gateway
        return {"NatGateway": nat_gateway}

    def describe_vpc_endpoints(self, *, Filters: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "VpcEndpoints": [
                item for item in self.vpc_endpoints.values() if _resource_matches_filters(item, Filters)
            ]
        }

    def create_vpc_endpoint(
        self,
        *,
        VpcId: str,
        ServiceName: str,
        VpcEndpointType: str,
        RouteTableIds: list[str],
        TagSpecifications: list[dict[str, Any]],
    ) -> dict[str, Any]:
        assert VpcEndpointType == "Gateway"
        endpoint_id = self._new_id("vpce")
        endpoint = {
            "RouteTableIds": RouteTableIds,
            "ServiceName": ServiceName,
            "Tags": TagSpecifications[0]["Tags"],
            "VpcEndpointId": endpoint_id,
            "VpcId": VpcId,
        }
        self.vpc_endpoints[endpoint_id] = endpoint
        return {"VpcEndpoint": endpoint}

    def modify_vpc_endpoint(self, *, VpcEndpointId: str, AddRouteTableIds: list[str]) -> None:
        existing = self.vpc_endpoints[VpcEndpointId]["RouteTableIds"]
        merged = sorted(set(existing) | set(AddRouteTableIds))
        self.vpc_endpoints[VpcEndpointId]["RouteTableIds"] = merged

    def describe_security_groups(self, *, Filters: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "SecurityGroups": [
                item for item in self.security_groups.values() if _resource_matches_filters(item, Filters)
            ]
        }

    def create_security_group(
        self,
        *,
        GroupName: str,
        Description: str,
        VpcId: str,
        TagSpecifications: list[dict[str, Any]],
    ) -> dict[str, Any]:
        group_id = self._new_id("sg")
        self.security_groups[group_id] = {
            "Description": Description,
            "GroupId": group_id,
            "GroupName": GroupName,
            "IpPermissions": [],
            "IpPermissionsEgress": [{"IpProtocol": "-1", "IpRanges": [{"CidrIp": "0.0.0.0/0"}]}],
            "Tags": TagSpecifications[0]["Tags"],
            "VpcId": VpcId,
        }
        return {"GroupId": group_id}

    def authorize_security_group_ingress(self, *, GroupId: str, IpPermissions: list[dict[str, Any]]) -> None:
        self.security_groups[GroupId]["IpPermissions"].extend(IpPermissions)

    def authorize_security_group_egress(self, *, GroupId: str, IpPermissions: list[dict[str, Any]]) -> None:
        self.security_groups[GroupId]["IpPermissionsEgress"].extend(IpPermissions)


class FakeSageMakerClient:
    def __init__(self, region_name: str) -> None:
        self.meta = FakeMeta(region_name)
        self.create_domain_attempts = 0
        self.create_domain_failures_remaining = 0
        self.domains: dict[str, dict[str, Any]] = {}
        self.user_profiles: dict[tuple[str, str], dict[str, Any]] = {}
        self.tags_by_arn: dict[str, list[dict[str, str]]] = {}
        self.update_domain_calls = 0
        self.update_domain_failures_remaining = 0
        self.update_user_profile_calls = 0
        self._next_domain = 1

    def list_domains(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "Domains": [
                {
                    "DomainId": item["DomainId"],
                    "DomainName": item["DomainName"],
                }
                for item in self.domains.values()
            ]
        }

    def create_domain(self, **kwargs: Any) -> dict[str, Any]:
        self.create_domain_attempts += 1
        if self.create_domain_failures_remaining > 0:
            self.create_domain_failures_remaining -= 1
            raise FakeAwsError(
                "ValidationException",
                (
                    "SageMaker is unable to perform: sts:AssumeRole on the role. "
                    "Check if you have a trust relationship for 'sagemaker.amazonaws.com' to assume the role."
                ),
            )
        domain_id = f"d-{self._next_domain:04d}"
        self._next_domain += 1
        domain_arn = f"arn:aws:sagemaker:{self.meta.region_name}:123456789012:domain/{domain_id}"
        domain = {
            "AppNetworkAccessType": kwargs["AppNetworkAccessType"],
            "AuthMode": kwargs["AuthMode"],
            "DefaultUserSettings": kwargs["DefaultUserSettings"],
            "DomainArn": domain_arn,
            "DomainId": domain_id,
            "DomainName": kwargs["DomainName"],
            "Status": "InService",
            "SubnetIds": kwargs["SubnetIds"],
            "VpcId": kwargs["VpcId"],
        }
        self.domains[domain_id] = domain
        self.tags_by_arn[domain_arn] = kwargs["Tags"]
        return {"DomainArn": domain_arn}

    def describe_domain(self, *, DomainId: str) -> dict[str, Any]:
        return self.domains[DomainId]

    def update_domain(self, *, DomainId: str, DefaultUserSettings: dict[str, Any], **_kwargs: Any) -> None:
        if self.update_domain_failures_remaining > 0:
            self.update_domain_failures_remaining -= 1
            raise FakeAwsError(
                "ValidationException",
                (
                    "SageMaker is unable to perform: sts:AssumeRole on the role. "
                    "Check if you have a trust relationship for 'sagemaker.amazonaws.com' to assume the role."
                ),
            )
        self.update_domain_calls += 1
        self.domains[DomainId]["DefaultUserSettings"] = DefaultUserSettings

    def list_tags(self, *, ResourceArn: str) -> dict[str, Any]:
        return {"Tags": self.tags_by_arn.get(ResourceArn, [])}

    def add_tags(self, *, ResourceArn: str, Tags: list[dict[str, str]]) -> None:
        current = _tags_to_dict(self.tags_by_arn.get(ResourceArn, []))
        current.update(_tags_to_dict(Tags))
        self.tags_by_arn[ResourceArn] = [{"Key": key, "Value": value} for key, value in sorted(current.items())]

    def list_user_profiles(self, *, DomainIdEquals: str, **_kwargs: Any) -> dict[str, Any]:
        return {
            "UserProfiles": [
                {
                    "DomainId": domain_id,
                    "UserProfileName": profile_name,
                }
                for (domain_id, profile_name), _profile in self.user_profiles.items()
                if domain_id == DomainIdEquals
            ]
        }

    def create_user_profile(self, **kwargs: Any) -> dict[str, Any]:
        domain_id = kwargs["DomainId"]
        profile_name = kwargs["UserProfileName"]
        user_profile_arn = (
            f"arn:aws:sagemaker:{self.meta.region_name}:123456789012:user-profile/{domain_id}/{profile_name}"
        )
        profile = {
            "DomainId": domain_id,
            "Status": "InService",
            "UserProfileArn": user_profile_arn,
            "UserProfileName": profile_name,
            "UserSettings": kwargs["UserSettings"],
        }
        if "SingleSignOnUserIdentifier" in kwargs:
            profile["SingleSignOnUserIdentifier"] = kwargs["SingleSignOnUserIdentifier"]
        if "SingleSignOnUserValue" in kwargs:
            profile["SingleSignOnUserValue"] = kwargs["SingleSignOnUserValue"]
        self.user_profiles[(domain_id, profile_name)] = profile
        self.tags_by_arn[user_profile_arn] = kwargs["Tags"]
        return {"UserProfileArn": user_profile_arn}

    def describe_user_profile(self, *, DomainId: str, UserProfileName: str) -> dict[str, Any]:
        return self.user_profiles[(DomainId, UserProfileName)]

    def update_user_profile(self, *, DomainId: str, UserProfileName: str, UserSettings: dict[str, Any]) -> None:
        self.update_user_profile_calls += 1
        self.user_profiles[(DomainId, UserProfileName)]["UserSettings"] = UserSettings


class FakeSession:
    def __init__(self, env: FakeAwsEnvironment) -> None:
        self.env = env

    def client(self, service_name: str) -> Any:
        return {
            "ec2": self.env.ec2,
            "iam": self.env.iam,
            "s3": self.env.s3,
            "sagemaker": self.env.sagemaker,
            "sso-admin": self.env.sso_admin,
        }[service_name]


class FakeSsoAdminClient:
    def __init__(self, region_name: str) -> None:
        self.meta = FakeMeta(region_name)
        self.instances = [{"InstanceArn": f"arn:aws:sso::123456789012:instance/{region_name}"}]

    def list_instances(self) -> dict[str, Any]:
        return {"Instances": list(self.instances)}


class FakeAwsEnvironment:
    def __init__(self, region_name: str = "us-east-2") -> None:
        self.region_name = region_name
        self.s3 = FakeS3Client(region_name)
        self.ec2 = FakeEc2Client(region_name)
        self.iam = FakeIamClient(region_name)
        self.sagemaker = FakeSageMakerClient(region_name)
        self.sso_admin = FakeSsoAdminClient(region_name)

    def session_factory(self, region_name: str) -> FakeSession:
        assert region_name == self.region_name
        return FakeSession(self)


def write_config(tmp_path: Path, *, users: list[dict[str, Any]] | None = None) -> Path:
    users = users or []
    lines = [
        "domain_name: lumina-studio",
        "bucket_name: ai4bio-lumina",
        "auth_mode: IAM",
        "tags:",
        "  environment: test",
        "network:",
        "  vpc_cidr: 10.32.0.0/16",
        "  private_subnet_cidrs:",
        "    - 10.32.0.0/20",
        "    - 10.32.16.0/20",
        "  public_subnet_cidr: 10.32.240.0/24",
        "  availability_zone_count: 2",
        "  nat_enabled: true",
        "execution_role:",
        "  name: lumina-sagemaker-execution-role",
        "  managed_policies:",
        "    - arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
        "  inline_bucket_access: true",
        "users:",
    ]
    for user in users:
        lines.append(f"  - profile_name: {user['profile_name']}")
        if "sso_username" in user:
            lines.append(f"    sso_username: {user['sso_username']}")
    config_path = tmp_path / "domain.yaml"
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path


def rewrite_auth_mode(config_path: Path, auth_mode: str) -> Path:
    updated = config_path.read_text(encoding="utf-8").replace("auth_mode: IAM", f"auth_mode: {auth_mode}")
    config_path.write_text(updated, encoding="utf-8")
    return config_path


def test_load_domain_config_applies_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("domain_name: demo-domain\n", encoding="utf-8")

    config = load_domain_config(config_path)

    assert config.domain_name == "demo-domain"
    assert config.bucket_name == "ai4bio-lumina"
    assert config.auth_mode == "IAM"
    assert config.network.nat_enabled is True
    assert config.execution_role.name == "lumina-sagemaker-execution-role"
    assert config.users == ()


def test_planned_ids_use_aws_like_prefixes() -> None:
    assert PLANNED_IDS["vpc"].startswith("vpc-")
    assert PLANNED_IDS["private-subnet-1"].startswith("subnet-")
    assert PLANNED_IDS["public-route-table"].startswith("rtb-")
    assert PLANNED_IDS["igw"].startswith("igw-")
    assert PLANNED_IDS["eip"].startswith("eipalloc-")
    assert PLANNED_IDS["nat"].startswith("nat-")
    assert PLANNED_IDS["s3-endpoint"].startswith("vpce-")
    assert PLANNED_IDS["security-group"].startswith("sg-")
    assert PLANNED_IDS["domain"].startswith("d-")


def test_apply_configuration_creates_bucket_and_domain_stack(tmp_path: Path) -> None:
    env = FakeAwsEnvironment()
    config = load_domain_config(
        write_config(tmp_path, users=[{"profile_name": "alice"}])
    )

    summary = apply_configuration(config, session_clients(env))

    assert "s3_bucket:ai4bio-lumina" in summary.created
    assert "sagemaker_domain:lumina-studio" in summary.created
    assert "user_profile:alice" in summary.created
    assert env.s3.buckets["ai4bio-lumina"]["VersioningStatus"] == "Enabled"
    assert len(env.ec2.vpcs) == 1
    assert len(env.ec2.subnets) == 3
    assert env.sagemaker.domains
    assert env.sagemaker.user_profiles


def test_domain_create_retries_role_propagation_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.sagemaker_domain_toolkit.time.sleep", lambda _seconds: None)
    env = FakeAwsEnvironment()
    env.sagemaker.create_domain_failures_remaining = 1
    config = load_domain_config(write_config(tmp_path))

    summary = apply_configuration(config, session_clients(env))

    assert env.sagemaker.create_domain_attempts == 2
    assert env.sagemaker.domains
    assert any("CreateDomain hit IAM propagation lag" in warning for warning in summary.warnings)


def test_domain_create_fails_fast_when_identity_center_is_missing(tmp_path: Path) -> None:
    env = FakeAwsEnvironment()
    env.sso_admin.instances = []
    config_path = write_config(tmp_path, users=[{"profile_name": "alice", "sso_username": "alice"}])
    config = load_domain_config(rewrite_auth_mode(config_path, "SSO"))

    with pytest.raises(RuntimeError, match="IAM Identity Center is not enabled"):
        apply_configuration(config, session_clients(env))


def test_sso_mode_requires_sso_usernames(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, users=[{"profile_name": "alice"}])

    with pytest.raises(ValueError, match="sso_username is required when auth_mode=SSO"):
        load_domain_config(rewrite_auth_mode(config_path, "SSO"))


def test_bucket_rejects_inaccessible_existing_bucket(tmp_path: Path) -> None:
    env = FakeAwsEnvironment()
    env.s3.inaccessible_buckets.add("ai4bio-lumina")
    config = load_domain_config(write_config(tmp_path))

    with pytest.raises(DriftError, match="not accessible"):
        apply_configuration(config, session_clients(env))


def test_network_and_domain_reruns_are_idempotent(tmp_path: Path) -> None:
    env = FakeAwsEnvironment()
    config = load_domain_config(
        write_config(tmp_path, users=[{"profile_name": "alice"}])
    )

    first_summary = apply_configuration(config, session_clients(env))
    second_summary = apply_configuration(config, session_clients(env))

    assert first_summary.created
    assert not second_summary.created
    assert not second_summary.updated
    assert "vpc:lumina-studio" in second_summary.unchanged
    assert "route_table:private-route-table" in second_summary.unchanged
    assert "nat_gateway:main" in second_summary.unchanged
    assert "sagemaker_domain:lumina-studio" in second_summary.unchanged
    assert "user_profile:alice" in second_summary.unchanged


def test_domain_updates_mutable_default_settings(tmp_path: Path) -> None:
    env = FakeAwsEnvironment()
    config = load_domain_config(write_config(tmp_path))

    apply_configuration(config, session_clients(env))
    domain_id = next(iter(env.sagemaker.domains))
    env.sagemaker.domains[domain_id]["DefaultUserSettings"] = {
        "ExecutionRole": "arn:aws:iam::123456789012:role/old-role",
        "SecurityGroups": ["sg-old"],
    }

    summary = apply_configuration(config, session_clients(env))

    assert "sagemaker_domain:lumina-studio" in summary.updated
    assert env.sagemaker.update_domain_calls == 1


def test_domain_rejects_immutable_drift(tmp_path: Path) -> None:
    env = FakeAwsEnvironment()
    config = load_domain_config(write_config(tmp_path))

    apply_configuration(config, session_clients(env))
    domain_id = next(iter(env.sagemaker.domains))
    env.sagemaker.domains[domain_id]["AuthMode"] = "SSO"

    with pytest.raises(DriftError, match="AuthMode"):
        apply_configuration(config, session_clients(env))


def test_existing_failed_domain_surfaces_failure_reason(tmp_path: Path) -> None:
    env = FakeAwsEnvironment()
    config = load_domain_config(write_config(tmp_path))
    domain_arn = env.sagemaker.create_domain(
        DomainName="lumina-studio",
        AuthMode="IAM",
        AppNetworkAccessType="VpcOnly",
        AppSecurityGroupManagement="Customer",
        DefaultUserSettings={"ExecutionRole": "arn:aws:iam::123456789012:role/demo", "SecurityGroups": ["sg-demo"]},
        SubnetIds=["subnet-a", "subnet-b"],
        VpcId="vpc-demo",
        Tags=[],
    )["DomainArn"]
    domain_id = domain_arn.rsplit("/", 1)[-1]
    env.sagemaker.domains[domain_id]["Status"] = "Failed"
    env.sagemaker.domains[domain_id]["FailureReason"] = "ConfigurationError: Enable AWS Single Sign-On in Region."

    with pytest.raises(RuntimeError, match="already in Failed state"):
        apply_configuration(config, session_clients(env))


def test_wait_for_domain_includes_failure_reason() -> None:
    class FailedDomainClient:
        def describe_domain(self, *, DomainId: str) -> dict[str, Any]:
            return {
                "DomainId": DomainId,
                "FailureReason": "ConfigurationError: Enable AWS Single Sign-On in Region.",
                "Status": "Failed",
            }

    with pytest.raises(RuntimeError, match="FailureReason"):
        _wait_for_domain(FailedDomainClient(), "d-demo", max_attempts=1, sleep_seconds=0.0)


def test_user_profile_updates_mutable_settings_and_rejects_username_mismatch(tmp_path: Path) -> None:
    env = FakeAwsEnvironment()
    config = load_domain_config(
        write_config(tmp_path, users=[{"profile_name": "alice"}])
    )

    apply_configuration(config, session_clients(env))
    domain_id = next(iter(env.sagemaker.domains))
    env.sagemaker.user_profiles[(domain_id, "alice")]["UserSettings"] = {
        "ExecutionRole": "arn:aws:iam::123456789012:role/other",
        "SecurityGroups": ["sg-other"],
    }

    summary = apply_configuration(config, session_clients(env))
    assert "user_profile:alice" in summary.updated
    assert env.sagemaker.update_user_profile_calls == 1


def test_sso_user_profile_rejects_username_mismatch(tmp_path: Path) -> None:
    env = FakeAwsEnvironment()
    config_path = write_config(tmp_path, users=[{"profile_name": "alice", "sso_username": "alice"}])
    config = load_domain_config(rewrite_auth_mode(config_path, "SSO"))

    apply_configuration(config, session_clients(env))
    domain_id = next(iter(env.sagemaker.domains))
    env.sagemaker.user_profiles[(domain_id, "alice")]["SingleSignOnUserValue"] = "bob"

    with pytest.raises(DriftError, match="expected 'alice'"):
        apply_configuration(config, session_clients(env))


def test_main_apply_dry_run_prints_json_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    env = FakeAwsEnvironment()
    config_path = write_config(tmp_path)

    exit_code = main_apply(
        ["--config", str(config_path), "--dry-run", "--region", env.region_name],
        session_factory=env.session_factory,
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["dry_run"] is True
    assert "s3_bucket:ai4bio-lumina" in payload["created"]
    assert not env.s3.buckets


def test_main_apply_returns_nonzero_on_drift_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    env = FakeAwsEnvironment()
    env.s3.inaccessible_buckets.add("ai4bio-lumina")
    config_path = write_config(tmp_path)

    exit_code = main_apply(
        ["--config", str(config_path), "--region", env.region_name],
        session_factory=env.session_factory,
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["error_type"] == "DriftError"
    assert "not accessible" in payload["error"]


def test_main_status_reports_drift_without_failing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    env = FakeAwsEnvironment()
    config_path = write_config(tmp_path)
    config = load_domain_config(config_path)

    apply_configuration(config, session_clients(env))
    domain_id = next(iter(env.sagemaker.domains))
    env.sagemaker.domains[domain_id]["AuthMode"] = "SSO"

    exit_code = main_status(
        ["--config", str(config_path), "--region", env.region_name],
        session_factory=env.session_factory,
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["drift"]
    assert "AuthMode" in payload["drift"][0]


def session_clients(env: FakeAwsEnvironment) -> AwsClients:
    return AwsClients(
        ec2=env.ec2,
        iam=env.iam,
        s3=env.s3,
        sagemaker=env.sagemaker,
        sso_admin=env.sso_admin,
    )
