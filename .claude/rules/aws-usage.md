# AWS usage

Budget is the team's AWS credits ($200 per participant; free tier ends if the account joins an AWS Organization — do not create one).

- Region: `us-east-1`.
- Always pass `--profile "$AWS_PROFILE"` (from `.env`). Never use the default profile, root credentials, or print/commit keys.
- Compute: a SageMaker Notebook Instance (`ml.t3.medium` is free-tier, 250 h). Train inside the notebook. Scale to a
  bigger instance or a Training Job only when a measured run is too slow, and say what it costs first.
- Never create SageMaker endpoints — the deliverable is TSV files, not an API, and endpoints bill while idle.
- Stop (don't delete) notebook instances when idle; confirm the stop.
- Data: keep a single S3 bucket for the dataset and artifacts. The dataset is organizer data — keep the bucket private.
- Before creating any billable resource (instance, GPU, training job, bucket), state the resource, instance type, and
  expected hourly cost, and get confirmation. List what's running before ending a work session.
- A billing alarm/budget should exist on the account before heavy use.
