# SageMaker Domain Provisioning

This repo includes an idempotent SageMaker Studio domain bootstrap toolkit under
[`aws/sagemaker-domain/`](../../aws/sagemaker-domain/).

It creates or reconciles:

- the `ai4bio-lumina` S3 bucket
- a dedicated VPC with two private subnets, one public subnet, NAT, route tables, and an S3 gateway endpoint
- a shared SageMaker execution role
- a SageMaker domain configured for `AuthMode=IAM` by default and `VpcOnly`
- declared SageMaker user profiles

The toolkit is intentionally imperative and Python-based so it matches the repo's existing SageMaker launchers and is
easy to unit test locally.

## Important Assumptions

- `auth_mode: IAM` is the default and does not require IAM Identity Center.
- If you switch to `auth_mode: SSO`, IAM Identity Center must already be enabled in the target region and any
  referenced usernames must already exist there.
- The toolkit does not create or manage IAM Identity Center users, groups, or assignments.
- Undeclared extra resources are reported, but not deleted.

## Config

The checked-in manifest lives at [`aws/sagemaker-domain/config.yaml`](../../aws/sagemaker-domain/config.yaml).

Key fields:

- `domain_name`
- `bucket_name`
- `auth_mode`
- `tags`
- `network`
- `execution_role`
- `users`

Each `users` entry declares:

- `profile_name`
- optional `sso_username` when `auth_mode: SSO`
- optional `execution_role_arn`
- optional `tags`

## Commands

Plan changes without mutating AWS:

```bash
uv run python aws/sagemaker-domain/apply.py --dry-run
```

Apply the desired state:

```bash
uv run python aws/sagemaker-domain/apply.py
```

Inspect the current discovered state and drift notes:

```bash
uv run python aws/sagemaker-domain/status.py
```

Both commands accept:

- `--config <path>`
- `--region <aws-region>`

The default region follows the existing SageMaker launcher convention:

- `AWS_DEFAULT_REGION`
- otherwise `us-east-2`

## Behavior Notes

- The bucket is created only if `ai4bio-lumina` is missing and accessible to the current credentials.
- If the bucket name already exists but is inaccessible, the run fails rather than attempting to replace it.
- The domain is created with `AppNetworkAccessType=VpcOnly` and a shared execution role.
- In `auth_mode: IAM`, user profiles are created without Single Sign-On bindings.
- In `auth_mode: SSO`, user profiles are created with `SingleSignOnUserIdentifier=UserName`.
- Immutable drift such as a mismatched auth mode, or in `SSO` mode a profile bound to the wrong IAM Identity Center
  username, causes apply runs to fail fast.
- `status.py` reports the same drift as JSON without mutating resources.

## Verification

Focused local verification for the toolkit:

```bash
uv run pytest tests/test_sagemaker_domain_toolkit.py
uv run ruff check src/sagemaker_domain_toolkit.py tests/test_sagemaker_domain_toolkit.py aws/sagemaker-domain/
uv run pyrefly check --summarize-errors
```
