from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml

from src.repo_paths import REPO_ROOT, resolve_repo_relative_path
from src.sagemaker_utils import load_dotenv_if_available

DEFAULT_REGION = "us-east-2"
DEFAULT_CONFIG_PATH = REPO_ROOT / "aws" / "sagemaker-domain" / "config.yaml"
DEFAULT_SAGEMAKER_POLICY_ARN = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
DEFAULT_ROLE_POLICY_NAME = "lumina-s3-access"
MANAGED_BY_TAG = "lumina-ssm"
STACK_TAG_VALUE = "sagemaker-domain"
IMMUTABLE_DRIFT_PREFIX = "immutable drift:"
WAIT_TERMINAL_STATUSES = {"InService"}
WAIT_FAILURE_STATUSES = {"Failed", "Delete_Failed", "Update_Failed"}
PLANNED_IDS = {
    "domain": "d-000000000000",
    "eip": "eipalloc-00000000000000000",
    "igw": "igw-00000000000000000",
    "nat": "nat-00000000000000000",
    "private-subnet-1": "subnet-00000000000000001",
    "private-subnet-2": "subnet-00000000000000002",
    "public-subnet": "subnet-00000000000000003",
    "public-route-table": "rtb-00000000000000001",
    "private-route-table": "rtb-00000000000000002",
    "security-group": "sg-00000000000000001",
    "s3-endpoint": "vpce-00000000000000001",
    "vpc": "vpc-00000000000000001",
}


class DriftError(RuntimeError):
    """Raised when the current AWS state is incompatible with the desired config."""


@dataclass(frozen=True)
class NetworkConfig:
    vpc_cidr: str = "10.32.0.0/16"
    private_subnet_cidrs: tuple[str, ...] = ("10.32.0.0/20", "10.32.16.0/20")
    public_subnet_cidr: str = "10.32.240.0/24"
    availability_zone_count: int = 2
    nat_enabled: bool = True


@dataclass(frozen=True)
class ExecutionRoleConfig:
    name: str = "lumina-sagemaker-execution-role"
    managed_policies: tuple[str, ...] = (DEFAULT_SAGEMAKER_POLICY_ARN,)
    inline_bucket_access: bool = True


@dataclass(frozen=True)
class UserProfileConfig:
    profile_name: str
    sso_username: str | None = None
    execution_role_arn: str | None = None
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DomainConfig:
    domain_name: str = "lumina-studio"
    bucket_name: str = "ai4bio-lumina"
    auth_mode: str = "IAM"
    tags: dict[str, str] = field(default_factory=dict)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    execution_role: ExecutionRoleConfig = field(default_factory=ExecutionRoleConfig)
    users: tuple[UserProfileConfig, ...] = ()


@dataclass
class ReconcileSummary:
    dry_run: bool = False
    created: set[str] = field(default_factory=set)
    updated: set[str] = field(default_factory=set)
    unchanged: set[str] = field(default_factory=set)
    warnings: set[str] = field(default_factory=set)
    drift: set[str] = field(default_factory=set)
    resolved: dict[str, Any] = field(default_factory=dict)
    current_state: dict[str, Any] = field(default_factory=dict)

    def mark_created(self, resource: str) -> None:
        self.created.add(resource)
        self.updated.discard(resource)
        self.unchanged.discard(resource)

    def mark_updated(self, resource: str) -> None:
        if resource not in self.created:
            self.updated.add(resource)
        self.unchanged.discard(resource)

    def mark_unchanged(self, resource: str) -> None:
        if resource not in self.created and resource not in self.updated:
            self.unchanged.add(resource)

    def add_warning(self, warning: str) -> None:
        self.warnings.add(warning)

    def add_drift(self, message: str) -> None:
        self.drift.add(message)

    def to_dict(self, *, error: str | None = None, error_type: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "created": sorted(self.created),
            "dry_run": self.dry_run,
            "resolved": self.resolved,
            "unchanged": sorted(self.unchanged),
            "updated": sorted(self.updated),
            "warnings": sorted(self.warnings),
        }
        if self.drift:
            payload["drift"] = sorted(self.drift)
        if self.current_state:
            payload["current_state"] = self.current_state
        if error is not None:
            payload["error"] = error
        if error_type is not None:
            payload["error_type"] = error_type
        return payload


@dataclass(frozen=True)
class AwsClients:
    s3: Any
    ec2: Any
    iam: Any
    sagemaker: Any
    sso_admin: Any


@dataclass(frozen=True)
class NetworkStack:
    vpc_id: str
    private_subnet_ids: tuple[str, ...]
    public_subnet_id: str
    private_route_table_id: str
    public_route_table_id: str
    internet_gateway_id: str
    nat_gateway_id: str | None
    eip_allocation_id: str | None
    s3_endpoint_id: str
    security_group_id: str


def load_domain_config(path: str | Path = DEFAULT_CONFIG_PATH) -> DomainConfig:
    resolved = resolve_repo_relative_path(path, repo_root=REPO_ROOT)
    if not resolved.is_file():
        raise FileNotFoundError(f"Domain config not found: {resolved}")
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping at top level of {resolved}")

    network_raw = cast(dict[str, Any], raw.get("network") or {})
    execution_role_raw = cast(dict[str, Any], raw.get("execution_role") or {})
    users_raw = raw.get("users") or []
    if not isinstance(users_raw, list):
        raise ValueError("`users` must be a list.")

    network = NetworkConfig(
        vpc_cidr=_require_string(network_raw.get("vpc_cidr", NetworkConfig.vpc_cidr), "network.vpc_cidr"),
        private_subnet_cidrs=_require_string_tuple(
            network_raw.get("private_subnet_cidrs", list(NetworkConfig.private_subnet_cidrs)),
            "network.private_subnet_cidrs",
        ),
        public_subnet_cidr=_require_string(
            network_raw.get("public_subnet_cidr", NetworkConfig.public_subnet_cidr),
            "network.public_subnet_cidr",
        ),
        availability_zone_count=_require_int(
            network_raw.get("availability_zone_count", NetworkConfig.availability_zone_count),
            "network.availability_zone_count",
        ),
        nat_enabled=_require_bool(network_raw.get("nat_enabled", NetworkConfig.nat_enabled), "network.nat_enabled"),
    )
    if network.availability_zone_count <= 0:
        raise ValueError("network.availability_zone_count must be greater than zero.")
    if len(network.private_subnet_cidrs) < network.availability_zone_count:
        raise ValueError("network.private_subnet_cidrs must cover network.availability_zone_count.")

    execution_role = ExecutionRoleConfig(
        name=_require_string(execution_role_raw.get("name", ExecutionRoleConfig.name), "execution_role.name"),
        managed_policies=_require_string_tuple(
            execution_role_raw.get("managed_policies", list(ExecutionRoleConfig.managed_policies)),
            "execution_role.managed_policies",
        ),
        inline_bucket_access=_require_bool(
            execution_role_raw.get("inline_bucket_access", ExecutionRoleConfig.inline_bucket_access),
            "execution_role.inline_bucket_access",
        ),
    )
    users = tuple(_load_user_config(item) for item in users_raw)
    auth_mode = _normalize_auth_mode(raw.get("auth_mode", DomainConfig.auth_mode))
    if auth_mode == "SSO":
        for user in users:
            if not user.sso_username:
                raise ValueError("users[].sso_username is required when auth_mode=SSO.")
    return DomainConfig(
        domain_name=_require_string(raw.get("domain_name", DomainConfig.domain_name), "domain_name"),
        bucket_name=_require_string(raw.get("bucket_name", DomainConfig.bucket_name), "bucket_name"),
        auth_mode=auth_mode,
        tags=_require_string_dict(raw.get("tags") or {}, "tags"),
        network=network,
        execution_role=execution_role,
        users=users,
    )


def _load_user_config(raw: Any) -> UserProfileConfig:
    if not isinstance(raw, dict):
        raise ValueError("Each item in `users` must be a mapping.")
    return UserProfileConfig(
        profile_name=_require_string(raw.get("profile_name"), "users[].profile_name"),
        sso_username=_optional_string(raw.get("sso_username"), "users[].sso_username"),
        execution_role_arn=_optional_string(raw.get("execution_role_arn"), "users[].execution_role_arn"),
        tags=_require_string_dict(raw.get("tags") or {}, "users[].tags"),
    )


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, field_name)


def _require_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    return value


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean.")
    return value


def _require_string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of strings.")
    resolved = tuple(_require_string(item, field_name) for item in value)
    if not resolved:
        raise ValueError(f"{field_name} must not be empty.")
    return resolved


def _require_string_dict(value: Any, field_name: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping of strings.")
    resolved: dict[str, str] = {}
    for key, item in value.items():
        resolved[_require_string(key, field_name)] = _require_string(item, field_name)
    return resolved


def _normalize_auth_mode(value: Any) -> str:
    normalized = _require_string(value, "auth_mode").upper()
    if normalized not in {"IAM", "SSO"}:
        raise ValueError("auth_mode must be either 'IAM' or 'SSO'.")
    return normalized


def build_aws_clients(
    region_name: str,
    *,
    session_factory: Any | None = None,
) -> AwsClients:
    session: Any
    if session_factory is None:
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError(
                "AWS provisioning dependencies are missing. Install them with `uv sync --extra sagemaker`."
            ) from exc
        session = boto3.Session(region_name=region_name)
    else:
        session = session_factory(region_name)
    return AwsClients(
        s3=session.client("s3"),
        ec2=session.client("ec2"),
        iam=session.client("iam"),
        sagemaker=session.client("sagemaker"),
        sso_admin=session.client("sso-admin"),
    )


def apply_configuration(
    config: DomainConfig,
    clients: AwsClients,
    *,
    dry_run: bool = False,
    fail_on_immutable_drift: bool = True,
) -> ReconcileSummary:
    summary = ReconcileSummary(dry_run=dry_run)
    _preflight_domain_prerequisites(config, clients, summary)
    bucket = ensure_bucket(config, clients.s3, summary, dry_run=dry_run)
    network = ensure_network_stack(
        config,
        clients.ec2,
        summary,
        dry_run=dry_run,
        fail_on_immutable_drift=fail_on_immutable_drift,
    )
    role = ensure_execution_role(config, clients.iam, summary, dry_run=dry_run)
    domain = ensure_sagemaker_domain(
        config,
        clients.sagemaker,
        summary,
        network=network,
        execution_role_arn=role["role_arn"],
        dry_run=dry_run,
        fail_on_immutable_drift=fail_on_immutable_drift,
    )
    ensure_user_profiles(
        config,
        clients.sagemaker,
        summary,
        domain_id=domain["domain_id"],
        security_group_id=network.security_group_id,
        default_execution_role_arn=role["role_arn"],
        dry_run=dry_run,
        fail_on_immutable_drift=fail_on_immutable_drift,
    )
    summary.resolved.setdefault("bucket", bucket)
    summary.resolved.setdefault("network", _network_stack_to_dict(network))
    summary.resolved.setdefault("execution_role", role)
    summary.resolved.setdefault("domain", domain)
    return summary


def _preflight_domain_prerequisites(
    config: DomainConfig,
    clients: AwsClients,
    summary: ReconcileSummary,
) -> None:
    existing_domain = _find_domain_by_name(clients.sagemaker, config.domain_name)
    if existing_domain is not None:
        domain_id = _require_present(existing_domain.get("DomainId"), "DomainId")
        described = clients.sagemaker.describe_domain(DomainId=domain_id)
        summary.current_state["domain"] = {
            "app_network_access_type": described.get("AppNetworkAccessType"),
            "auth_mode": described.get("AuthMode"),
            "domain_arn": described.get("DomainArn"),
            "domain_id": domain_id,
            "domain_name": described.get("DomainName"),
            "failure_reason": described.get("FailureReason"),
            "status": described.get("Status"),
            "subnet_ids": described.get("SubnetIds", []),
            "vpc_id": described.get("VpcId"),
        }
        if described.get("Status") == "Failed":
            failure_reason = described.get("FailureReason") or "Unknown failure reason."
            raise RuntimeError(
                f"SageMaker domain {domain_id} is already in Failed state. "
                f"FailureReason: {failure_reason} "
                "Delete the failed domain after fixing the underlying issue, then rerun apply."
            )
        return
    if config.auth_mode == "SSO":
        _ensure_identity_center_enabled(clients.sso_admin)


def ensure_bucket(config: DomainConfig, s3: Any, summary: ReconcileSummary, *, dry_run: bool) -> dict[str, Any]:
    bucket_name = config.bucket_name
    resource_label = f"s3_bucket:{bucket_name}"
    access_state = _bucket_access_state(s3, bucket_name)
    bucket_state: dict[str, Any] = {"name": bucket_name, "state": access_state}
    summary.current_state["bucket"] = bucket_state

    if access_state == "forbidden":
        raise DriftError(
            f"S3 bucket {bucket_name!r} exists but is not accessible to the current credentials. "
            "Refusing to reuse a foreign-owned or inaccessible bucket."
        )

    if access_state == "missing":
        summary.mark_created(resource_label)
        if not dry_run:
            create_kwargs: dict[str, Any] = {"Bucket": bucket_name}
            region_name = _client_region(s3)
            if region_name != "us-east-1":
                create_kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region_name}
            s3.create_bucket(**create_kwargs)
        bucket_state["state"] = "created" if not dry_run else "planned-create"
    else:
        summary.mark_unchanged(resource_label)

    if access_state != "missing" or not dry_run:
        _ensure_bucket_versioning(s3, bucket_name, summary, resource_label, dry_run=dry_run)
        _ensure_bucket_encryption(s3, bucket_name, summary, resource_label, dry_run=dry_run)
        _ensure_bucket_public_access_block(s3, bucket_name, summary, resource_label, dry_run=dry_run)
        _ensure_bucket_tags(config, s3, bucket_name, summary, resource_label, dry_run=dry_run)

    bucket_location = _normalize_bucket_region(_safe_get_bucket_location(s3, bucket_name))
    if bucket_location and bucket_location != _client_region(s3):
        summary.add_warning(
            f"S3 bucket {bucket_name!r} lives in region {bucket_location}, "
            f"while the toolkit is targeting {_client_region(s3)}."
        )
    bucket_arn = f"arn:aws:s3:::{bucket_name}"
    resolved = {
        "bucket_arn": bucket_arn,
        "bucket_name": bucket_name,
        "bucket_region": bucket_location or _client_region(s3),
    }
    summary.resolved["bucket"] = resolved
    summary.current_state["bucket"] = {**bucket_state, **resolved}
    return resolved


def ensure_network_stack(
    config: DomainConfig,
    ec2: Any,
    summary: ReconcileSummary,
    *,
    dry_run: bool,
    fail_on_immutable_drift: bool,
) -> NetworkStack:
    azs = _available_azs(ec2, config.network.availability_zone_count)

    vpc = _find_tagged_resource(
        ec2.describe_vpcs(
            Filters=_managed_filters(config.domain_name, "vpc"),
        ).get("Vpcs", []),
        "VPC",
        config.domain_name,
    )
    vpc_label = f"vpc:{config.domain_name}"
    if vpc is None:
        summary.mark_created(vpc_label)
        if dry_run:
            vpc_id = PLANNED_IDS["vpc"]
            vpc = {"VpcId": vpc_id, "CidrBlock": config.network.vpc_cidr, "Tags": []}
        else:
            create_response = ec2.create_vpc(
                CidrBlock=config.network.vpc_cidr,
                TagSpecifications=_tag_specifications(
                    "vpc",
                    _component_tags(config, "vpc", name=f"{config.domain_name}-vpc"),
                ),
            )
            vpc = create_response["Vpc"]
            ec2.modify_vpc_attribute(VpcId=vpc["VpcId"], EnableDnsSupport={"Value": True})
            ec2.modify_vpc_attribute(VpcId=vpc["VpcId"], EnableDnsHostnames={"Value": True})
    else:
        vpc_id = _require_present(vpc.get("VpcId"), "VpcId")
        _immutable_check(
            summary,
            fail_on_immutable_drift,
            vpc.get("CidrBlock") == config.network.vpc_cidr,
            f"{IMMUTABLE_DRIFT_PREFIX} managed VPC {vpc_id} has CIDR {vpc.get('CidrBlock')},"
            f" expected {config.network.vpc_cidr}.",
        )
        _ensure_ec2_tags(
            ec2,
            vpc_id,
            _component_tags(config, "vpc", name=f"{config.domain_name}-vpc"),
            current_tags=_tag_dict(vpc.get("Tags")),
            summary=summary,
            resource_label=vpc_label,
            dry_run=dry_run,
        )
        summary.mark_unchanged(vpc_label)
    vpc_id = _require_present(vpc.get("VpcId"), "VpcId")

    private_subnet_ids: list[str] = []
    subnet_state: list[dict[str, Any]] = []
    for index in range(config.network.availability_zone_count):
        component = f"private-subnet-{index + 1}"
        subnet_label = f"subnet:{component}"
        desired_cidr = config.network.private_subnet_cidrs[index]
        desired_az = azs[index]
        subnet = _find_tagged_resource(
            ec2.describe_subnets(
                Filters=[
                    {"Name": "vpc-id", "Values": [vpc_id]},
                    *_managed_filters(config.domain_name, component),
                ]
            ).get("Subnets", []),
            "subnet",
            component,
        )
        if subnet is None:
            summary.mark_created(subnet_label)
            if dry_run:
                subnet = {
                    "AvailabilityZone": desired_az,
                    "CidrBlock": desired_cidr,
                    "SubnetId": PLANNED_IDS[component],
                    "Tags": [],
                }
            else:
                response = ec2.create_subnet(
                    VpcId=vpc_id,
                    CidrBlock=desired_cidr,
                    AvailabilityZone=desired_az,
                    TagSpecifications=_tag_specifications(
                        "subnet",
                        _component_tags(config, component, name=f"{config.domain_name}-private-{index + 1}"),
                    ),
                )
                subnet = response["Subnet"]
                ec2.modify_subnet_attribute(SubnetId=subnet["SubnetId"], MapPublicIpOnLaunch={"Value": False})
        else:
            subnet_id = _require_present(subnet.get("SubnetId"), "SubnetId")
            _immutable_check(
                summary,
                fail_on_immutable_drift,
                subnet.get("CidrBlock") == desired_cidr,
                f"{IMMUTABLE_DRIFT_PREFIX} subnet {subnet_id} has CIDR "
                f"{subnet.get('CidrBlock')}, expected {desired_cidr}.",
            )
            _immutable_check(
                summary,
                fail_on_immutable_drift,
                subnet.get("AvailabilityZone") == desired_az,
                f"{IMMUTABLE_DRIFT_PREFIX} subnet {subnet_id} is in "
                f"{subnet.get('AvailabilityZone')}, expected {desired_az}.",
            )
            _ensure_ec2_tags(
                ec2,
                subnet_id,
                _component_tags(config, component, name=f"{config.domain_name}-private-{index + 1}"),
                current_tags=_tag_dict(subnet.get("Tags")),
                summary=summary,
                resource_label=subnet_label,
                dry_run=dry_run,
            )
            summary.mark_unchanged(subnet_label)
        subnet_id = _require_present(subnet.get("SubnetId"), "SubnetId")
        private_subnet_ids.append(subnet_id)
        subnet_state.append(
            {
                "availability_zone": desired_az,
                "cidr": desired_cidr,
                "subnet_id": subnet_id,
            }
        )

    public_subnet_component = "public-subnet"
    public_subnet_label = f"subnet:{public_subnet_component}"
    public_subnet = _find_tagged_resource(
        ec2.describe_subnets(
            Filters=[
                {"Name": "vpc-id", "Values": [vpc_id]},
                *_managed_filters(config.domain_name, public_subnet_component),
            ]
        ).get("Subnets", []),
        "subnet",
        public_subnet_component,
    )
    public_az = azs[0]
    if public_subnet is None:
        summary.mark_created(public_subnet_label)
        if dry_run:
            public_subnet = {
                "AvailabilityZone": public_az,
                "CidrBlock": config.network.public_subnet_cidr,
                "SubnetId": PLANNED_IDS["public-subnet"],
                "Tags": [],
            }
        else:
            response = ec2.create_subnet(
                VpcId=vpc_id,
                CidrBlock=config.network.public_subnet_cidr,
                AvailabilityZone=public_az,
                TagSpecifications=_tag_specifications(
                    "subnet",
                    _component_tags(config, public_subnet_component, name=f"{config.domain_name}-public-1"),
                ),
            )
            public_subnet = response["Subnet"]
            ec2.modify_subnet_attribute(SubnetId=public_subnet["SubnetId"], MapPublicIpOnLaunch={"Value": True})
    else:
        public_subnet_id = _require_present(public_subnet.get("SubnetId"), "SubnetId")
        _immutable_check(
            summary,
            fail_on_immutable_drift,
            public_subnet.get("CidrBlock") == config.network.public_subnet_cidr,
            f"{IMMUTABLE_DRIFT_PREFIX} public subnet {public_subnet_id} has CIDR {public_subnet.get('CidrBlock')},"
            f" expected {config.network.public_subnet_cidr}.",
        )
        _immutable_check(
            summary,
            fail_on_immutable_drift,
            public_subnet.get("AvailabilityZone") == public_az,
            f"{IMMUTABLE_DRIFT_PREFIX} public subnet {public_subnet_id} is in {public_subnet.get('AvailabilityZone')},"
            f" expected {public_az}.",
        )
        _ensure_ec2_tags(
            ec2,
            public_subnet_id,
            _component_tags(config, public_subnet_component, name=f"{config.domain_name}-public-1"),
            current_tags=_tag_dict(public_subnet.get("Tags")),
            summary=summary,
            resource_label=public_subnet_label,
            dry_run=dry_run,
        )
        summary.mark_unchanged(public_subnet_label)
    public_subnet_id = _require_present(public_subnet.get("SubnetId"), "SubnetId")

    internet_gateway = _ensure_internet_gateway(
        config,
        ec2,
        summary,
        vpc_id=vpc_id,
        dry_run=dry_run,
    )
    public_route_table = _ensure_route_table(
        config,
        ec2,
        summary,
        vpc_id=vpc_id,
        component="public-route-table",
        name=f"{config.domain_name}-public-rt",
        dry_run=dry_run,
    )
    _ensure_default_route(
        ec2,
        summary,
        route_table=public_route_table,
        route_table_label="route_table:public-route-table",
        gateway_id=_require_present(internet_gateway.get("InternetGatewayId"), "InternetGatewayId"),
        dry_run=dry_run,
    )
    _ensure_route_table_association(
        ec2,
        summary,
        vpc_id=vpc_id,
        route_table_id=_require_present(public_route_table.get("RouteTableId"), "RouteTableId"),
        subnet_id=public_subnet_id,
        label="route_assoc:public-subnet",
        dry_run=dry_run,
    )

    nat_gateway_id: str | None = None
    eip_allocation_id: str | None = None
    if config.network.nat_enabled:
        eip = _ensure_eip(config, ec2, summary, dry_run=dry_run)
        eip_allocation_id = _require_present(eip.get("AllocationId"), "AllocationId")
        nat_gateway = _ensure_nat_gateway(
            config,
            ec2,
            summary,
            vpc_id=vpc_id,
            public_subnet_id=public_subnet_id,
            allocation_id=eip_allocation_id,
            dry_run=dry_run,
            fail_on_immutable_drift=fail_on_immutable_drift,
        )
        nat_gateway_id = _require_present(nat_gateway.get("NatGatewayId"), "NatGatewayId")
        if not dry_run:
            _wait_for_nat_gateway(ec2, nat_gateway_id)
    else:
        summary.add_warning("NAT is disabled, so Studio apps will not have general outbound internet access.")

    private_route_table = _ensure_route_table(
        config,
        ec2,
        summary,
        vpc_id=vpc_id,
        component="private-route-table",
        name=f"{config.domain_name}-private-rt",
        dry_run=dry_run,
    )
    if nat_gateway_id is not None:
        _ensure_default_route(
            ec2,
            summary,
            route_table=private_route_table,
            route_table_label="route_table:private-route-table",
            nat_gateway_id=nat_gateway_id,
            dry_run=dry_run,
        )
    for index, subnet_id in enumerate(private_subnet_ids):
        _ensure_route_table_association(
            ec2,
            summary,
            vpc_id=vpc_id,
            route_table_id=_require_present(private_route_table.get("RouteTableId"), "RouteTableId"),
            subnet_id=subnet_id,
            label=f"route_assoc:private-subnet-{index + 1}",
            dry_run=dry_run,
        )

    s3_endpoint = _ensure_s3_gateway_endpoint(
        config,
        ec2,
        summary,
        vpc_id=vpc_id,
        route_table_id=_require_present(private_route_table.get("RouteTableId"), "RouteTableId"),
        dry_run=dry_run,
    )
    security_group = _ensure_security_group(
        config,
        ec2,
        summary,
        vpc_id=vpc_id,
        dry_run=dry_run,
        fail_on_immutable_drift=fail_on_immutable_drift,
    )

    stack = NetworkStack(
        vpc_id=vpc_id,
        private_subnet_ids=tuple(private_subnet_ids),
        public_subnet_id=public_subnet_id,
        private_route_table_id=_require_present(private_route_table.get("RouteTableId"), "RouteTableId"),
        public_route_table_id=_require_present(public_route_table.get("RouteTableId"), "RouteTableId"),
        internet_gateway_id=_require_present(internet_gateway.get("InternetGatewayId"), "InternetGatewayId"),
        nat_gateway_id=nat_gateway_id,
        eip_allocation_id=eip_allocation_id,
        s3_endpoint_id=_require_present(s3_endpoint.get("VpcEndpointId"), "VpcEndpointId"),
        security_group_id=_require_present(security_group.get("GroupId"), "GroupId"),
    )
    summary.current_state["network"] = {
        "private_subnets": subnet_state,
        "public_subnet_id": public_subnet_id,
        "vpc_id": vpc_id,
    }
    summary.resolved["network"] = _network_stack_to_dict(stack)
    return stack


def ensure_execution_role(
    config: DomainConfig,
    iam: Any,
    summary: ReconcileSummary,
    *,
    dry_run: bool,
) -> dict[str, str]:
    role_name = config.execution_role.name
    resource_label = f"iam_role:{role_name}"
    role = None
    try:
        role = iam.get_role(RoleName=role_name)["Role"]
    except Exception as exc:
        if _aws_error_code(exc) not in {"NoSuchEntity", "NoSuchEntityException"}:
            raise
    desired_tags = _component_tags(config, "execution-role", name=role_name)
    if role is None:
        summary.mark_created(resource_label)
        role_arn = f"arn:aws:iam::000000000000:role/{role_name}"
        if not dry_run:
            response = iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(_sagemaker_assume_role_policy(), sort_keys=True),
                Description=f"Shared SageMaker execution role for {config.domain_name}",
                Tags=_aws_tags(desired_tags),
            )
            role = response["Role"]
            role_arn = _require_present(role.get("Arn"), "Arn")
    else:
        role_arn = _require_present(role.get("Arn"), "Arn")
        current_tags = _tag_dict(role.get("Tags"))
        if not _has_desired_tags(current_tags, desired_tags):
            summary.mark_updated(resource_label)
            if not dry_run:
                iam.tag_role(RoleName=role_name, Tags=_aws_tags(desired_tags))
        else:
            summary.mark_unchanged(resource_label)
        assume_policy = json.dumps(_sagemaker_assume_role_policy(), sort_keys=True)
        current_assume_policy = json.dumps(role.get("AssumeRolePolicyDocument") or {}, sort_keys=True)
        if current_assume_policy != assume_policy:
            summary.mark_updated(resource_label)
            if not dry_run:
                iam.update_assume_role_policy(RoleName=role_name, PolicyDocument=assume_policy)

    if role is not None or not dry_run:
        attached = iam.list_attached_role_policies(RoleName=role_name).get("AttachedPolicies", [])
        attached_arns = {item.get("PolicyArn") for item in attached}
        for policy_arn in config.execution_role.managed_policies:
            if policy_arn not in attached_arns:
                summary.mark_updated(resource_label)
                if not dry_run:
                    iam.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)

        if config.execution_role.inline_bucket_access:
            desired_policy = json.dumps(_bucket_inline_policy(config.bucket_name), sort_keys=True)
            current_policy = None
            try:
                response = iam.get_role_policy(RoleName=role_name, PolicyName=DEFAULT_ROLE_POLICY_NAME)
                current_policy = json.dumps(response.get("PolicyDocument") or {}, sort_keys=True)
            except Exception as exc:
                if _aws_error_code(exc) not in {"NoSuchEntity", "NoSuchEntityException"}:
                    raise
            if current_policy != desired_policy:
                summary.mark_updated(resource_label)
                if not dry_run:
                    iam.put_role_policy(
                        RoleName=role_name,
                        PolicyName=DEFAULT_ROLE_POLICY_NAME,
                        PolicyDocument=desired_policy,
                    )

    resolved = {"role_arn": role_arn, "role_name": role_name}
    summary.current_state["execution_role"] = resolved
    summary.resolved["execution_role"] = resolved
    return resolved


def ensure_sagemaker_domain(
    config: DomainConfig,
    sagemaker: Any,
    summary: ReconcileSummary,
    *,
    network: NetworkStack,
    execution_role_arn: str,
    dry_run: bool,
    fail_on_immutable_drift: bool,
) -> dict[str, Any]:
    resource_label = f"sagemaker_domain:{config.domain_name}"
    domain = _find_domain_by_name(sagemaker, config.domain_name)
    desired_default_user_settings = {
        "ExecutionRole": execution_role_arn,
        "SecurityGroups": [network.security_group_id],
    }
    if domain is None:
        summary.mark_created(resource_label)
        domain_id = PLANNED_IDS["domain"]
        domain_arn = f"arn:aws:sagemaker:{_client_region(sagemaker)}:000000000000:domain/{domain_id}"
        if not dry_run:
            response = cast(
                dict[str, Any],
                _run_sagemaker_domain_role_retry(
                    summary,
                    operation_name="CreateDomain",
                    func=lambda: sagemaker.create_domain(
                        DomainName=config.domain_name,
                        AuthMode=config.auth_mode,
                        AppNetworkAccessType="VpcOnly",
                        AppSecurityGroupManagement="Customer",
                        DefaultUserSettings=desired_default_user_settings,
                        SubnetIds=list(network.private_subnet_ids),
                        VpcId=network.vpc_id,
                        Tags=_aws_tags(_component_tags(config, "domain", name=config.domain_name)),
                    ),
                ),
            )
            domain_arn = _require_present(response.get("DomainArn"), "DomainArn")
            domain_id = _domain_id_from_arn(domain_arn)
            _wait_for_domain(sagemaker, domain_id)
        resolved = {"domain_arn": domain_arn, "domain_id": domain_id, "domain_name": config.domain_name}
        summary.current_state["domain"] = {
            **resolved,
            "auth_mode": config.auth_mode,
            "app_network_access_type": "VpcOnly",
        }
        summary.resolved["domain"] = resolved
        return resolved

    domain_id = _require_present(domain.get("DomainId"), "DomainId")
    described = sagemaker.describe_domain(DomainId=domain_id)
    summary.current_state["domain"] = {
        "app_network_access_type": described.get("AppNetworkAccessType"),
        "auth_mode": described.get("AuthMode"),
        "domain_arn": described.get("DomainArn"),
        "domain_id": domain_id,
        "domain_name": described.get("DomainName"),
        "subnet_ids": described.get("SubnetIds", []),
        "vpc_id": described.get("VpcId"),
    }
    _immutable_check(
        summary,
        fail_on_immutable_drift,
        described.get("AuthMode") == config.auth_mode,
        f"{IMMUTABLE_DRIFT_PREFIX} SageMaker domain {config.domain_name!r} uses "
        f"AuthMode={described.get('AuthMode')!r}, expected {config.auth_mode!r}.",
    )
    _immutable_check(
        summary,
        fail_on_immutable_drift,
        described.get("AppNetworkAccessType") == "VpcOnly",
        f"{IMMUTABLE_DRIFT_PREFIX} SageMaker domain {config.domain_name!r} uses"
        f" AppNetworkAccessType={described.get('AppNetworkAccessType')!r}, expected 'VpcOnly'.",
    )
    _immutable_check(
        summary,
        fail_on_immutable_drift,
        described.get("VpcId") == network.vpc_id,
        f"{IMMUTABLE_DRIFT_PREFIX} SageMaker domain {config.domain_name!r} is attached to VPC"
        f" {described.get('VpcId')!r}, expected {network.vpc_id!r}.",
    )
    described_subnets = tuple(sorted(cast(list[str], described.get("SubnetIds") or [])))
    desired_subnets = tuple(sorted(network.private_subnet_ids))
    _immutable_check(
        summary,
        fail_on_immutable_drift,
        described_subnets == desired_subnets,
        f"{IMMUTABLE_DRIFT_PREFIX} SageMaker domain {config.domain_name!r} uses subnets {described_subnets!r},"
        f" expected {desired_subnets!r}.",
    )

    current_default_user_settings = cast(dict[str, Any], described.get("DefaultUserSettings") or {})
    needs_update = False
    if current_default_user_settings.get("ExecutionRole") != execution_role_arn:
        needs_update = True
    if tuple(sorted(cast(list[str], current_default_user_settings.get("SecurityGroups") or []))) != tuple(
        sorted(desired_default_user_settings["SecurityGroups"])
    ):
        needs_update = True

    desired_tags = _component_tags(config, "domain", name=config.domain_name)
    domain_arn = _require_present(described.get("DomainArn"), "DomainArn")
    _ensure_sagemaker_tags(
        sagemaker,
        domain_arn,
        desired_tags,
        summary,
        resource_label=resource_label,
        dry_run=dry_run,
    )

    if needs_update:
        summary.mark_updated(resource_label)
        if not dry_run:
            _run_sagemaker_domain_role_retry(
                summary,
                operation_name="UpdateDomain",
                func=lambda: sagemaker.update_domain(
                    DomainId=domain_id,
                    AppSecurityGroupManagement="Customer",
                    DefaultUserSettings=desired_default_user_settings,
                ),
            )
            _wait_for_domain(sagemaker, domain_id)
    else:
        summary.mark_unchanged(resource_label)
    resolved = {
        "domain_arn": domain_arn,
        "domain_id": domain_id,
        "domain_name": config.domain_name,
    }
    summary.resolved["domain"] = resolved
    return resolved


def ensure_user_profiles(
    config: DomainConfig,
    sagemaker: Any,
    summary: ReconcileSummary,
    *,
    domain_id: str,
    security_group_id: str,
    default_execution_role_arn: str,
    dry_run: bool,
    fail_on_immutable_drift: bool,
) -> None:
    listed_profiles = _list_user_profiles(sagemaker, domain_id)
    existing_by_name = {item["UserProfileName"]: item for item in listed_profiles}
    declared_names = {user.profile_name for user in config.users}
    extra_names = sorted(set(existing_by_name) - declared_names)
    for extra_name in extra_names:
        summary.add_warning(
            f"User profile {extra_name!r} exists in domain {domain_id} but is not declared in the manifest."
        )

    profile_resolved: dict[str, dict[str, str]] = {}
    profile_state: dict[str, dict[str, Any]] = {}
    for user in config.users:
        resource_label = f"user_profile:{user.profile_name}"
        execution_role_arn = user.execution_role_arn or default_execution_role_arn
        desired_user_settings = {
            "ExecutionRole": execution_role_arn,
            "SecurityGroups": [security_group_id],
        }
        existing = existing_by_name.get(user.profile_name)
        if existing is None:
            summary.mark_created(resource_label)
            user_profile_arn = (
                f"arn:aws:sagemaker:{_client_region(sagemaker)}:000000000000:"
                f"user-profile/{domain_id}/{user.profile_name}"
            )
            if not dry_run:
                user_tags = _component_tags(
                    config,
                    "user-profile",
                    name=user.profile_name,
                    extra_tags=user.tags,
                )
                create_kwargs: dict[str, Any] = {
                    "DomainId": domain_id,
                    "UserProfileName": user.profile_name,
                    "UserSettings": desired_user_settings,
                    "Tags": _aws_tags(user_tags),
                }
                if config.auth_mode == "SSO":
                    create_kwargs["SingleSignOnUserIdentifier"] = "UserName"
                    create_kwargs["SingleSignOnUserValue"] = user.sso_username
                response = sagemaker.create_user_profile(**create_kwargs)
                user_profile_arn = _require_present(response.get("UserProfileArn"), "UserProfileArn")
                _wait_for_user_profile(sagemaker, domain_id, user.profile_name)
            profile_resolved[user.profile_name] = {"user_profile_arn": user_profile_arn}
            if user.sso_username:
                profile_resolved[user.profile_name]["sso_username"] = user.sso_username
            profile_state[user.profile_name] = {
                "execution_role_arn": execution_role_arn,
                "status": "planned-create" if dry_run else "created",
            }
            if user.sso_username:
                profile_state[user.profile_name]["sso_username"] = user.sso_username
            continue

        described = sagemaker.describe_user_profile(DomainId=domain_id, UserProfileName=user.profile_name)
        profile_state[user.profile_name] = {
            "execution_role_arn": cast(dict[str, Any], described.get("UserSettings") or {}).get("ExecutionRole"),
            "status": described.get("Status"),
        }
        if described.get("SingleSignOnUserValue") is not None:
            profile_state[user.profile_name]["sso_username"] = described.get("SingleSignOnUserValue")
        if config.auth_mode == "SSO":
            _immutable_check(
                summary,
                fail_on_immutable_drift,
                described.get("SingleSignOnUserIdentifier") == "UserName",
                f"{IMMUTABLE_DRIFT_PREFIX} user profile {user.profile_name!r} is not bound by UserName.",
            )
            _immutable_check(
                summary,
                fail_on_immutable_drift,
                described.get("SingleSignOnUserValue") == user.sso_username,
                f"{IMMUTABLE_DRIFT_PREFIX} user profile {user.profile_name!r} targets"
                f" {described.get('SingleSignOnUserValue')!r}, expected {user.sso_username!r}.",
            )

        current_user_settings = cast(dict[str, Any], described.get("UserSettings") or {})
        needs_update = False
        if current_user_settings.get("ExecutionRole") != execution_role_arn:
            needs_update = True
        if tuple(sorted(cast(list[str], current_user_settings.get("SecurityGroups") or []))) != (security_group_id,):
            needs_update = True

        user_profile_arn = _require_present(described.get("UserProfileArn"), "UserProfileArn")
        _ensure_sagemaker_tags(
            sagemaker,
            user_profile_arn,
            _component_tags(config, "user-profile", name=user.profile_name, extra_tags=user.tags),
            summary,
            resource_label=resource_label,
            dry_run=dry_run,
        )

        if needs_update:
            summary.mark_updated(resource_label)
            if not dry_run:
                sagemaker.update_user_profile(
                    DomainId=domain_id,
                    UserProfileName=user.profile_name,
                    UserSettings=desired_user_settings,
                )
                _wait_for_user_profile(sagemaker, domain_id, user.profile_name)
        else:
            summary.mark_unchanged(resource_label)
        profile_resolved[user.profile_name] = {"user_profile_arn": user_profile_arn}
        if config.auth_mode == "SSO" and user.sso_username:
            profile_resolved[user.profile_name]["sso_username"] = user.sso_username

    summary.resolved["user_profiles"] = profile_resolved
    summary.current_state["user_profiles"] = profile_state


def build_apply_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create or reconcile the Lumina SageMaker domain stack.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to the domain YAML config (default: {DEFAULT_CONFIG_PATH.relative_to(REPO_ROOT)}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan changes without mutating AWS resources.",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=os.environ.get("AWS_DEFAULT_REGION", DEFAULT_REGION),
        help=f"AWS region (default: AWS_DEFAULT_REGION or {DEFAULT_REGION}).",
    )
    return parser


def build_status_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect the current Lumina SageMaker domain stack state.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to the domain YAML config (default: {DEFAULT_CONFIG_PATH.relative_to(REPO_ROOT)}).",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=os.environ.get("AWS_DEFAULT_REGION", DEFAULT_REGION),
        help=f"AWS region (default: AWS_DEFAULT_REGION or {DEFAULT_REGION}).",
    )
    return parser


def main_apply(argv: list[str] | None = None, *, session_factory: Any | None = None) -> int:
    load_dotenv_if_available()
    args = build_apply_arg_parser().parse_args(argv)
    summary = ReconcileSummary(dry_run=args.dry_run)
    try:
        config = load_domain_config(args.config)
        clients = build_aws_clients(args.region, session_factory=session_factory)
        summary = apply_configuration(config, clients, dry_run=args.dry_run, fail_on_immutable_drift=True)
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                summary.to_dict(error=str(exc), error_type=type(exc).__name__),
                indent=2,
                sort_keys=True,
            )
        )
        return 1


def main_status(argv: list[str] | None = None, *, session_factory: Any | None = None) -> int:
    load_dotenv_if_available()
    args = build_status_arg_parser().parse_args(argv)
    summary = ReconcileSummary(dry_run=True)
    try:
        config = load_domain_config(args.config)
        clients = build_aws_clients(args.region, session_factory=session_factory)
        summary = apply_configuration(config, clients, dry_run=True, fail_on_immutable_drift=False)
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                summary.to_dict(error=str(exc), error_type=type(exc).__name__),
                indent=2,
                sort_keys=True,
            )
        )
        return 1


def _managed_tags(config: DomainConfig) -> dict[str, str]:
    return {
        **config.tags,
        "lumina:domain-name": config.domain_name,
        "lumina:stack": STACK_TAG_VALUE,
        "managed-by": MANAGED_BY_TAG,
    }


def _component_tags(
    config: DomainConfig,
    component: str,
    *,
    name: str | None = None,
    extra_tags: dict[str, str] | None = None,
) -> dict[str, str]:
    tags = {
        **_managed_tags(config),
        "lumina:component": component,
    }
    if name is not None:
        tags["Name"] = name
    if extra_tags:
        tags.update(extra_tags)
    return tags


def _aws_tags(tags: dict[str, str]) -> list[dict[str, str]]:
    return [{"Key": key, "Value": value} for key, value in sorted(tags.items())]


def _tag_specifications(resource_type: str, tags: dict[str, str]) -> list[dict[str, Any]]:
    return [{"ResourceType": resource_type, "Tags": _aws_tags(tags)}]


def _managed_filters(domain_name: str, component: str) -> list[dict[str, Any]]:
    return [
        {"Name": "tag:managed-by", "Values": [MANAGED_BY_TAG]},
        {"Name": "tag:lumina:stack", "Values": [STACK_TAG_VALUE]},
        {"Name": "tag:lumina:domain-name", "Values": [domain_name]},
        {"Name": "tag:lumina:component", "Values": [component]},
    ]


def _tag_dict(tag_items: Any) -> dict[str, str]:
    if not isinstance(tag_items, list):
        return {}
    resolved: dict[str, str] = {}
    for item in tag_items:
        if not isinstance(item, dict):
            continue
        key = item.get("Key")
        value = item.get("Value")
        if isinstance(key, str) and isinstance(value, str):
            resolved[key] = value
    return resolved


def _available_azs(ec2: Any, count: int) -> list[str]:
    azs = [
        item["ZoneName"]
        for item in ec2.describe_availability_zones(Filters=[{"Name": "state", "Values": ["available"]}]).get(
            "AvailabilityZones",
            [],
        )
        if isinstance(item, dict) and isinstance(item.get("ZoneName"), str)
    ]
    if len(azs) < count:
        raise RuntimeError(f"Only found {len(azs)} availability zones, but {count} are required.")
    return azs[:count]


def _find_tagged_resource(items: list[dict[str, Any]], resource_type: str, component: str) -> dict[str, Any] | None:
    if not items:
        return None
    if len(items) > 1:
        raise DriftError(f"Found multiple managed {resource_type} resources for component {component!r}.")
    return items[0]


def _ensure_ec2_tags(
    ec2: Any,
    resource_id: str,
    desired_tags: dict[str, str],
    *,
    current_tags: dict[str, str],
    summary: ReconcileSummary,
    resource_label: str,
    dry_run: bool,
) -> bool:
    if _has_desired_tags(current_tags, desired_tags):
        return False
    summary.mark_updated(resource_label)
    if not dry_run:
        ec2.create_tags(Resources=[resource_id], Tags=_aws_tags(desired_tags))
    return True


def _ensure_internet_gateway(
    config: DomainConfig,
    ec2: Any,
    summary: ReconcileSummary,
    *,
    vpc_id: str,
    dry_run: bool,
) -> dict[str, Any]:
    resource_label = "internet_gateway:main"
    gateway = _find_tagged_resource(
        ec2.describe_internet_gateways(
            Filters=[
                {"Name": "attachment.vpc-id", "Values": [vpc_id]},
                *_managed_filters(config.domain_name, "internet-gateway"),
            ]
        ).get("InternetGateways", []),
        "internet gateway",
        "internet-gateway",
    )
    if gateway is None:
        summary.mark_created(resource_label)
        if dry_run:
            gateway = {
                "InternetGatewayId": PLANNED_IDS["igw"],
                "Attachments": [{"VpcId": vpc_id}],
                "Tags": [],
            }
        else:
            response = ec2.create_internet_gateway(
                TagSpecifications=_tag_specifications(
                    "internet-gateway",
                    _component_tags(config, "internet-gateway", name=f"{config.domain_name}-igw"),
                )
            )
            gateway = response["InternetGateway"]
            ec2.attach_internet_gateway(InternetGatewayId=gateway["InternetGatewayId"], VpcId=vpc_id)
    else:
        _ensure_ec2_tags(
            ec2,
            _require_present(gateway.get("InternetGatewayId"), "InternetGatewayId"),
            _component_tags(config, "internet-gateway", name=f"{config.domain_name}-igw"),
            current_tags=_tag_dict(gateway.get("Tags")),
            summary=summary,
            resource_label=resource_label,
            dry_run=dry_run,
        )
        summary.mark_unchanged(resource_label)
    return gateway


def _ensure_route_table(
    config: DomainConfig,
    ec2: Any,
    summary: ReconcileSummary,
    *,
    vpc_id: str,
    component: str,
    name: str,
    dry_run: bool,
) -> dict[str, Any]:
    resource_label = f"route_table:{component}"
    route_table = _find_tagged_resource(
        ec2.describe_route_tables(
            Filters=[
                {"Name": "vpc-id", "Values": [vpc_id]},
                *_managed_filters(config.domain_name, component),
            ]
        ).get("RouteTables", []),
        "route table",
        component,
    )
    if route_table is None:
        summary.mark_created(resource_label)
        if dry_run:
            route_table = {
                "RouteTableId": PLANNED_IDS[component],
                "Routes": [],
                "Associations": [],
                "Tags": [],
            }
        else:
            response = ec2.create_route_table(
                VpcId=vpc_id,
                TagSpecifications=_tag_specifications("route-table", _component_tags(config, component, name=name)),
            )
            route_table = response["RouteTable"]
    else:
        _ensure_ec2_tags(
            ec2,
            _require_present(route_table.get("RouteTableId"), "RouteTableId"),
            _component_tags(config, component, name=name),
            current_tags=_tag_dict(route_table.get("Tags")),
            summary=summary,
            resource_label=resource_label,
            dry_run=dry_run,
        )
        summary.mark_unchanged(resource_label)
    return route_table


def _ensure_default_route(
    ec2: Any,
    summary: ReconcileSummary,
    *,
    route_table: dict[str, Any],
    route_table_label: str,
    gateway_id: str | None = None,
    nat_gateway_id: str | None = None,
    dry_run: bool,
) -> None:
    route_table_id = _require_present(route_table.get("RouteTableId"), "RouteTableId")
    routes = cast(list[dict[str, Any]], route_table.get("Routes") or [])
    existing_route = next((route for route in routes if route.get("DestinationCidrBlock") == "0.0.0.0/0"), None)
    desired_matches = False
    if existing_route is not None:
        if gateway_id is not None and existing_route.get("GatewayId") == gateway_id:
            desired_matches = True
        if nat_gateway_id is not None and existing_route.get("NatGatewayId") == nat_gateway_id:
            desired_matches = True
    if desired_matches:
        summary.mark_unchanged(route_table_label)
        return
    summary.mark_updated(route_table_label)
    if dry_run:
        return
    if existing_route is None:
        kwargs = {"RouteTableId": route_table_id, "DestinationCidrBlock": "0.0.0.0/0"}
        if gateway_id is not None:
            kwargs["GatewayId"] = gateway_id
        if nat_gateway_id is not None:
            kwargs["NatGatewayId"] = nat_gateway_id
        ec2.create_route(**kwargs)
        return
    kwargs = {"RouteTableId": route_table_id, "DestinationCidrBlock": "0.0.0.0/0"}
    if gateway_id is not None:
        kwargs["GatewayId"] = gateway_id
    if nat_gateway_id is not None:
        kwargs["NatGatewayId"] = nat_gateway_id
    ec2.replace_route(**kwargs)


def _ensure_route_table_association(
    ec2: Any,
    summary: ReconcileSummary,
    *,
    vpc_id: str,
    route_table_id: str,
    subnet_id: str,
    label: str,
    dry_run: bool,
) -> None:
    route_tables = ec2.describe_route_tables(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}]).get("RouteTables", [])
    current_association: dict[str, Any] | None = None
    current_route_table_id: str | None = None
    for route_table in route_tables:
        if not isinstance(route_table, dict):
            continue
        for association in cast(list[dict[str, Any]], route_table.get("Associations") or []):
            if association.get("SubnetId") == subnet_id:
                current_association = association
                current_route_table_id = cast(str | None, route_table.get("RouteTableId"))
                break
        if current_association is not None:
            break
    if current_route_table_id == route_table_id:
        summary.mark_unchanged(label)
        return
    summary.mark_updated(label)
    if dry_run:
        return
    association_id = (
        cast(str | None, current_association.get("RouteTableAssociationId"))
        if current_association
        else None
    )
    if association_id:
        ec2.replace_route_table_association(AssociationId=association_id, RouteTableId=route_table_id)
    else:
        ec2.associate_route_table(RouteTableId=route_table_id, SubnetId=subnet_id)


def _ensure_eip(config: DomainConfig, ec2: Any, summary: ReconcileSummary, *, dry_run: bool) -> dict[str, Any]:
    resource_label = "elastic_ip:nat"
    addresses = ec2.describe_addresses(Filters=_managed_filters(config.domain_name, "nat-eip")).get("Addresses", [])
    address = _find_tagged_resource(addresses, "elastic IP", "nat-eip")
    if address is None:
        summary.mark_created(resource_label)
        if dry_run:
            return {"AllocationId": PLANNED_IDS["eip"]}
        address = ec2.allocate_address(Domain="vpc")
        ec2.create_tags(
            Resources=[address["AllocationId"]],
            Tags=_aws_tags(_component_tags(config, "nat-eip", name=f"{config.domain_name}-nat-eip")),
        )
        return address
    summary.mark_unchanged(resource_label)
    return address


def _ensure_nat_gateway(
    config: DomainConfig,
    ec2: Any,
    summary: ReconcileSummary,
    *,
    vpc_id: str,
    public_subnet_id: str,
    allocation_id: str,
    dry_run: bool,
    fail_on_immutable_drift: bool,
) -> dict[str, Any]:
    resource_label = "nat_gateway:main"
    gateways = ec2.describe_nat_gateways(
        Filter=[
            {"Name": "vpc-id", "Values": [vpc_id]},
            {"Name": "state", "Values": ["available", "pending"]},
            *_managed_filters(config.domain_name, "nat-gateway"),
        ]
    ).get("NatGateways", [])
    nat_gateway = _find_tagged_resource(gateways, "NAT gateway", "nat-gateway")
    if nat_gateway is None:
        summary.mark_created(resource_label)
        if dry_run:
            return {
                "NatGatewayAddresses": [{"AllocationId": allocation_id}],
                "NatGatewayId": PLANNED_IDS["nat"],
                "SubnetId": public_subnet_id,
            }
        response = ec2.create_nat_gateway(
            SubnetId=public_subnet_id,
            AllocationId=allocation_id,
            TagSpecifications=_tag_specifications(
                "natgateway",
                _component_tags(config, "nat-gateway", name=f"{config.domain_name}-nat"),
            ),
        )
        return response["NatGateway"]
    nat_gateway_id = _require_present(nat_gateway.get("NatGatewayId"), "NatGatewayId")
    _immutable_check(
        summary,
        fail_on_immutable_drift,
        nat_gateway.get("SubnetId") == public_subnet_id,
        f"{IMMUTABLE_DRIFT_PREFIX} NAT gateway {nat_gateway_id} is attached to subnet {nat_gateway.get('SubnetId')!r},"
        f" expected {public_subnet_id!r}.",
    )
    allocation_ids = {
        address.get("AllocationId")
        for address in cast(list[dict[str, Any]], nat_gateway.get("NatGatewayAddresses") or [])
        if isinstance(address, dict)
    }
    _immutable_check(
        summary,
        fail_on_immutable_drift,
        allocation_id in allocation_ids,
        f"{IMMUTABLE_DRIFT_PREFIX} NAT gateway {nat_gateway_id} is not using EIP allocation {allocation_id!r}.",
    )
    summary.mark_unchanged(resource_label)
    return nat_gateway


def _wait_for_nat_gateway(ec2: Any, nat_gateway_id: str, *, max_attempts: int = 40, sleep_seconds: float = 5.0) -> None:
    for _ in range(max_attempts):
        response = ec2.describe_nat_gateways(NatGatewayIds=[nat_gateway_id])
        gateways = cast(list[dict[str, Any]], response.get("NatGateways") or [])
        if not gateways:
            raise RuntimeError(f"NAT gateway {nat_gateway_id} disappeared while waiting for it to become available.")
        state = cast(str | None, gateways[0].get("State"))
        if state == "available":
            return
        if state in {"failed", "deleting", "deleted"}:
            raise RuntimeError(f"NAT gateway {nat_gateway_id} entered terminal state {state!r}.")
        time.sleep(sleep_seconds)
    raise RuntimeError(f"Timed out waiting for NAT gateway {nat_gateway_id} to become available.")


def _ensure_s3_gateway_endpoint(
    config: DomainConfig,
    ec2: Any,
    summary: ReconcileSummary,
    *,
    vpc_id: str,
    route_table_id: str,
    dry_run: bool,
) -> dict[str, Any]:
    resource_label = "vpc_endpoint:s3"
    service_name = f"com.amazonaws.{_client_region(ec2)}.s3"
    endpoints = ec2.describe_vpc_endpoints(
        Filters=[
            {"Name": "vpc-id", "Values": [vpc_id]},
            {"Name": "service-name", "Values": [service_name]},
            *_managed_filters(config.domain_name, "s3-endpoint"),
        ]
    ).get("VpcEndpoints", [])
    endpoint = _find_tagged_resource(endpoints, "VPC endpoint", "s3-endpoint")
    if endpoint is None:
        summary.mark_created(resource_label)
        if dry_run:
            return {
                "RouteTableIds": [route_table_id],
                "VpcEndpointId": PLANNED_IDS["s3-endpoint"],
            }
        response = ec2.create_vpc_endpoint(
            VpcId=vpc_id,
            ServiceName=service_name,
            VpcEndpointType="Gateway",
            RouteTableIds=[route_table_id],
            TagSpecifications=_tag_specifications(
                "vpc-endpoint",
                _component_tags(config, "s3-endpoint", name=f"{config.domain_name}-s3-endpoint"),
            ),
        )
        return cast(dict[str, Any], response.get("VpcEndpoint") or response)

    current_route_tables = set(cast(list[str], endpoint.get("RouteTableIds") or []))
    if route_table_id not in current_route_tables:
        summary.mark_updated(resource_label)
        if not dry_run:
            ec2.modify_vpc_endpoint(VpcEndpointId=endpoint["VpcEndpointId"], AddRouteTableIds=[route_table_id])
    else:
        summary.mark_unchanged(resource_label)
    return endpoint


def _ensure_security_group(
    config: DomainConfig,
    ec2: Any,
    summary: ReconcileSummary,
    *,
    vpc_id: str,
    dry_run: bool,
    fail_on_immutable_drift: bool,
) -> dict[str, Any]:
    resource_label = "security_group:studio"
    groups = ec2.describe_security_groups(
        Filters=[
            {"Name": "vpc-id", "Values": [vpc_id]},
            *_managed_filters(config.domain_name, "studio-security-group"),
        ]
    ).get("SecurityGroups", [])
    group = _find_tagged_resource(groups, "security group", "studio-security-group")
    if group is None:
        summary.mark_created(resource_label)
        if dry_run:
            group = {
                "GroupId": PLANNED_IDS["security-group"],
                "IpPermissions": [],
                "IpPermissionsEgress": [{"IpProtocol": "-1", "IpRanges": [{"CidrIp": "0.0.0.0/0"}]}],
                "Tags": [],
            }
        else:
            response = ec2.create_security_group(
                GroupName=f"{config.domain_name}-studio",
                Description=f"SageMaker Studio security group for {config.domain_name}",
                VpcId=vpc_id,
                TagSpecifications=_tag_specifications(
                    "security-group",
                    _component_tags(config, "studio-security-group", name=f"{config.domain_name}-studio"),
                ),
            )
            group_id = response["GroupId"]
            group = {
                "GroupId": group_id,
                "IpPermissions": [],
                "IpPermissionsEgress": [{"IpProtocol": "-1", "IpRanges": [{"CidrIp": "0.0.0.0/0"}]}],
                "Tags": [],
            }
    else:
        group_id = _require_present(group.get("GroupId"), "GroupId")
        _immutable_check(
            summary,
            fail_on_immutable_drift,
            group.get("VpcId") == vpc_id,
            f"{IMMUTABLE_DRIFT_PREFIX} security group {group_id} belongs to VPC "
            f"{group.get('VpcId')!r}, expected {vpc_id!r}.",
        )
        _ensure_ec2_tags(
            ec2,
            group_id,
            _component_tags(config, "studio-security-group", name=f"{config.domain_name}-studio"),
            current_tags=_tag_dict(group.get("Tags")),
            summary=summary,
            resource_label=resource_label,
            dry_run=dry_run,
        )
        summary.mark_unchanged(resource_label)
    group_id = _require_present(group.get("GroupId"), "GroupId")

    if not _security_group_has_nfs_self_rule(group):
        summary.mark_updated(resource_label)
        if not dry_run:
            ec2.authorize_security_group_ingress(
                GroupId=group_id,
                IpPermissions=[
                    {
                        "FromPort": 2049,
                        "IpProtocol": "tcp",
                        "ToPort": 2049,
                        "UserIdGroupPairs": [{"GroupId": group_id}],
                    }
                ],
            )

    if not _security_group_has_open_egress(group):
        summary.mark_updated(resource_label)
        if not dry_run:
            ec2.authorize_security_group_egress(
                GroupId=group_id,
                IpPermissions=[
                    {
                        "IpProtocol": "-1",
                        "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
                    }
                ],
            )
    return group


def _security_group_has_nfs_self_rule(group: dict[str, Any]) -> bool:
    for permission in cast(list[dict[str, Any]], group.get("IpPermissions") or []):
        if permission.get("IpProtocol") != "tcp":
            continue
        if permission.get("FromPort") != 2049 or permission.get("ToPort") != 2049:
            continue
        for pair in cast(list[dict[str, Any]], permission.get("UserIdGroupPairs") or []):
            if pair.get("GroupId") == group.get("GroupId"):
                return True
    return False


def _security_group_has_open_egress(group: dict[str, Any]) -> bool:
    for permission in cast(list[dict[str, Any]], group.get("IpPermissionsEgress") or []):
        if permission.get("IpProtocol") != "-1":
            continue
        for item in cast(list[dict[str, Any]], permission.get("IpRanges") or []):
            if item.get("CidrIp") == "0.0.0.0/0":
                return True
    return False


def _ensure_sagemaker_tags(
    sagemaker: Any,
    resource_arn: str,
    desired_tags: dict[str, str],
    summary: ReconcileSummary,
    *,
    resource_label: str,
    dry_run: bool,
) -> bool:
    current_tags = _tag_dict(sagemaker.list_tags(ResourceArn=resource_arn).get("Tags", []))
    if _has_desired_tags(current_tags, desired_tags):
        return False
    summary.mark_updated(resource_label)
    if not dry_run:
        sagemaker.add_tags(ResourceArn=resource_arn, Tags=_aws_tags(desired_tags))
    return True


def _ensure_identity_center_enabled(sso_admin: Any) -> None:
    response = sso_admin.list_instances()
    instances = cast(list[dict[str, Any]], response.get("Instances") or [])
    if instances:
        return
    raise RuntimeError(
        f"IAM Identity Center is not enabled in region {_client_region(sso_admin)!r}. "
        "SageMaker domains with AuthMode=SSO require Identity Center in the same region."
    )


def _run_sagemaker_domain_role_retry(
    summary: ReconcileSummary,
    *,
    operation_name: str,
    func: Callable[[], Any],
    max_attempts: int = 6,
    sleep_seconds: float = 10.0,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except Exception as exc:
            if not _is_sagemaker_assume_role_propagation_error(exc):
                raise
            last_error = exc
            if attempt == max_attempts:
                raise
            summary.add_warning(
                f"{operation_name} hit IAM propagation lag for the SageMaker execution role; "
                f"retrying ({attempt}/{max_attempts - 1})."
            )
            time.sleep(sleep_seconds)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"{operation_name} failed before execution started.")


def _wait_for_domain(sagemaker: Any, domain_id: str, *, max_attempts: int = 80, sleep_seconds: float = 15.0) -> None:
    for _ in range(max_attempts):
        described = sagemaker.describe_domain(DomainId=domain_id)
        status = cast(str | None, described.get("Status"))
        if status in WAIT_TERMINAL_STATUSES:
            return
        if status in WAIT_FAILURE_STATUSES:
            failure_reason = cast(str | None, described.get("FailureReason"))
            if failure_reason:
                raise RuntimeError(
                    f"SageMaker domain {domain_id} entered terminal state {status!r}. "
                    f"FailureReason: {failure_reason}"
                )
            raise RuntimeError(f"SageMaker domain {domain_id} entered terminal state {status!r}.")
        time.sleep(sleep_seconds)
    raise RuntimeError(f"Timed out waiting for SageMaker domain {domain_id} to enter service.")


def _wait_for_user_profile(
    sagemaker: Any,
    domain_id: str,
    user_profile_name: str,
    *,
    max_attempts: int = 40,
    sleep_seconds: float = 10.0,
) -> None:
    for _ in range(max_attempts):
        described = sagemaker.describe_user_profile(DomainId=domain_id, UserProfileName=user_profile_name)
        status = cast(str | None, described.get("Status"))
        if status in WAIT_TERMINAL_STATUSES:
            return
        if status in WAIT_FAILURE_STATUSES:
            raise RuntimeError(
                f"User profile {user_profile_name!r} in domain {domain_id} entered terminal state {status!r}."
            )
        time.sleep(sleep_seconds)
    raise RuntimeError(f"Timed out waiting for user profile {user_profile_name!r} to enter service.")


def _find_domain_by_name(sagemaker: Any, domain_name: str) -> dict[str, Any] | None:
    matches = [item for item in _list_domains(sagemaker) if item.get("DomainName") == domain_name]
    if not matches:
        return None
    if len(matches) > 1:
        raise DriftError(f"Found multiple SageMaker domains named {domain_name!r}.")
    return matches[0]


def _list_domains(sagemaker: Any) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    next_token: str | None = None
    while True:
        kwargs: dict[str, Any] = {}
        if next_token:
            kwargs["NextToken"] = next_token
        response = sagemaker.list_domains(**kwargs)
        for item in cast(list[dict[str, Any]], response.get("Domains") or []):
            matches.append(item)
        next_token = cast(str | None, response.get("NextToken"))
        if not next_token:
            return matches


def _list_user_profiles(sagemaker: Any, domain_id: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    next_token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"DomainIdEquals": domain_id}
        if next_token:
            kwargs["NextToken"] = next_token
        response = sagemaker.list_user_profiles(**kwargs)
        for item in cast(list[dict[str, Any]], response.get("UserProfiles") or []):
            matches.append(item)
        next_token = cast(str | None, response.get("NextToken"))
        if not next_token:
            return matches


def _bucket_access_state(s3: Any, bucket_name: str) -> str:
    try:
        s3.head_bucket(Bucket=bucket_name)
        return "accessible"
    except Exception as exc:
        code = _aws_error_code(exc)
        status_code = _aws_status_code(exc)
        if code in {"NoSuchBucket", "404", "NotFound"} or status_code == 404:
            return "missing"
        if code in {"403", "AccessDenied", "AllAccessDisabled"} or status_code == 403:
            return "forbidden"
        raise


def _ensure_bucket_versioning(
    s3: Any,
    bucket_name: str,
    summary: ReconcileSummary,
    resource_label: str,
    *,
    dry_run: bool,
) -> None:
    current = s3.get_bucket_versioning(Bucket=bucket_name).get("Status")
    if current == "Enabled":
        return
    summary.mark_updated(resource_label)
    if not dry_run:
        s3.put_bucket_versioning(Bucket=bucket_name, VersioningConfiguration={"Status": "Enabled"})


def _ensure_bucket_encryption(
    s3: Any,
    bucket_name: str,
    summary: ReconcileSummary,
    resource_label: str,
    *,
    dry_run: bool,
) -> None:
    desired = {
        "Rules": [
            {
                "ApplyServerSideEncryptionByDefault": {
                    "SSEAlgorithm": "AES256",
                }
            }
        ]
    }
    current = None
    try:
        current = s3.get_bucket_encryption(Bucket=bucket_name).get("ServerSideEncryptionConfiguration")
    except Exception as exc:
        if _aws_error_code(exc) not in {"ServerSideEncryptionConfigurationNotFoundError"}:
            raise
    if current == desired:
        return
    summary.mark_updated(resource_label)
    if not dry_run:
        s3.put_bucket_encryption(Bucket=bucket_name, ServerSideEncryptionConfiguration=desired)


def _ensure_bucket_public_access_block(
    s3: Any,
    bucket_name: str,
    summary: ReconcileSummary,
    resource_label: str,
    *,
    dry_run: bool,
) -> None:
    desired = {
        "BlockPublicAcls": True,
        "BlockPublicPolicy": True,
        "IgnorePublicAcls": True,
        "RestrictPublicBuckets": True,
    }
    current = None
    try:
        current = s3.get_public_access_block(Bucket=bucket_name).get("PublicAccessBlockConfiguration")
    except Exception as exc:
        if _aws_error_code(exc) not in {"NoSuchPublicAccessBlockConfiguration"}:
            raise
    if current == desired:
        return
    summary.mark_updated(resource_label)
    if not dry_run:
        s3.put_public_access_block(Bucket=bucket_name, PublicAccessBlockConfiguration=desired)


def _ensure_bucket_tags(
    config: DomainConfig,
    s3: Any,
    bucket_name: str,
    summary: ReconcileSummary,
    resource_label: str,
    *,
    dry_run: bool,
) -> None:
    desired_tags = _component_tags(config, "bucket", name=bucket_name)
    current = {}
    try:
        current = _tag_dict(s3.get_bucket_tagging(Bucket=bucket_name).get("TagSet", []))
    except Exception as exc:
        if _aws_error_code(exc) not in {"NoSuchTagSet"}:
            raise
    if current == desired_tags:
        return
    summary.mark_updated(resource_label)
    if not dry_run:
        s3.put_bucket_tagging(Bucket=bucket_name, Tagging={"TagSet": _aws_tags(desired_tags)})


def _safe_get_bucket_location(s3: Any, bucket_name: str) -> str | None:
    try:
        return cast(str | None, s3.get_bucket_location(Bucket=bucket_name).get("LocationConstraint"))
    except Exception:
        return None


def _normalize_bucket_region(location: str | None) -> str | None:
    if location in {None, ""}:
        return "us-east-1"
    return location


def _client_region(client: Any) -> str:
    meta = getattr(client, "meta", None)
    region_name = getattr(meta, "region_name", None)
    if isinstance(region_name, str) and region_name:
        return region_name
    return DEFAULT_REGION


def _aws_error_code(exc: Exception) -> str | None:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error")
        if isinstance(error, dict):
            code = error.get("Code")
            if isinstance(code, str):
                return code
    return None


def _aws_status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        metadata = response.get("ResponseMetadata")
        if isinstance(metadata, dict):
            status_code = metadata.get("HTTPStatusCode")
            if isinstance(status_code, int):
                return status_code
    return None


def _is_sagemaker_assume_role_propagation_error(exc: Exception) -> bool:
    code = _aws_error_code(exc)
    if code != "ValidationException":
        return False
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    error = response.get("Error")
    if not isinstance(error, dict):
        return False
    message = error.get("Message")
    if not isinstance(message, str):
        return False
    normalized = message.lower()
    return "sts:assumerole" in normalized and "trust relationship" in normalized


def _require_present(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Expected {field_name} to be present.")
    return value


def _immutable_check(
    summary: ReconcileSummary,
    fail_on_immutable_drift: bool,
    condition: bool,
    message: str,
) -> None:
    if condition:
        return
    summary.add_drift(message)
    if fail_on_immutable_drift:
        raise DriftError(message)


def _domain_id_from_arn(domain_arn: str) -> str:
    return domain_arn.rsplit("/", 1)[-1]


def _network_stack_to_dict(stack: NetworkStack) -> dict[str, Any]:
    return {
        "eip_allocation_id": stack.eip_allocation_id,
        "internet_gateway_id": stack.internet_gateway_id,
        "nat_gateway_id": stack.nat_gateway_id,
        "private_route_table_id": stack.private_route_table_id,
        "private_subnet_ids": list(stack.private_subnet_ids),
        "public_route_table_id": stack.public_route_table_id,
        "public_subnet_id": stack.public_subnet_id,
        "s3_endpoint_id": stack.s3_endpoint_id,
        "security_group_id": stack.security_group_id,
        "vpc_id": stack.vpc_id,
    }


def _sagemaker_assume_role_policy() -> dict[str, Any]:
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }


def _bucket_inline_policy(bucket_name: str) -> dict[str, Any]:
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["s3:ListBucket"],
                "Resource": [f"arn:aws:s3:::{bucket_name}"],
            },
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject",
                ],
                "Resource": [f"arn:aws:s3:::{bucket_name}/*"],
            },
        ],
    }


def _has_desired_tags(current_tags: dict[str, str], desired_tags: dict[str, str]) -> bool:
    return all(current_tags.get(key) == value for key, value in desired_tags.items())
