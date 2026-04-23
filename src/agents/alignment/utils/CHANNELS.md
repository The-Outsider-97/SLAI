# Configuring Notification Channels for Human Oversight

This document provides detailed instructions for setting up the two primary notification channels used by the `HumanOversightInterface` module: **Slack Incoming Webhooks** and **SMTP email**.

## 1. Slack Incoming Webhook

A Slack Incoming Webhook allows the alignment system to post intervention requests directly into a designated Slack channel. The following steps assume you have administrator access to a Slack workspace.

### 1.1 Create a Slack Application

1. Navigate to [https://api.slack.com/apps](https://api.slack.com/apps) in a web browser.
2. Click the **"Create New App"** button.
3. In the modal window, select **"From scratch"**.
4. Enter an **App Name** – for example, `AI Alignment Oversight`.
5. Select the **workspace** where you want the alerts to appear from the dropdown menu.
6. Click **"Create App"**.

### 1.2 Activate the Incoming Webhook Feature

1. After the app is created, you are redirected to its **"Basic Information"** page.
2. In the left sidebar, locate the **"Features"** section and click **"Incoming Webhooks"**.
3. At the top of the page, toggle the switch from **"Off"** to **"On"**.
   - When enabled, a new section labelled **"Webhook URLs for Your Workspace"** appears below.

### 1.3 Generate a Webhook URL

1. In the **"Webhook URLs for Your Workspace"** section, click the **"Add New Webhook to Workspace"** button.
2. A Slack permission dialog opens. Select the **channel** where you wish to receive the intervention alerts (e.g., `#alignment-alerts`).
3. Click **"Allow"**.
4. The page refreshes, and the new webhook URL appears in the list. The URL has the following format:
   ```
   https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX
   ```
5. Click the **"Copy"** button next to the URL.

### 1.4 Security Considerations

- The webhook URL functions as a secret token. Anyone possessing the URL can post messages to the configured channel.
- **Do not** hardcode the URL in configuration files that are committed to version control.
- In production, store the webhook URL in a secure secrets manager (e.g., HashiCorp Vault, AWS Secrets Manager, Azure Key Vault) and load it at runtime via environment variables or a secure configuration endpoint.

### 1.5 Configuration in `alignment_config.yaml`

Insert the copied URL into the `human_oversight.channels.slack.webhook_url` field:

```yaml
human_oversight:
  channels:
    slack:
      webhook_url: "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX"
      retry_count: 3
```

The `retry_count` parameter defines how many times the system attempts to deliver a notification before marking it as failed.

---

## 2. SMTP Email Server

The SMTP (Simple Mail Transfer Protocol) email channel is used to send intervention alerts to designated reviewers. The exact server settings depend on your organisation's email infrastructure or the third‑party email service you choose.

### 2.1 Choose an SMTP Service

For production environments, a dedicated email delivery service is recommended. Common options include:

| Service      | SMTP Server              | Port (TLS) | Authentication          |
|--------------|--------------------------|------------|-------------------------|
| SendGrid     | `smtp.sendgrid.net`      | 587        | API key as username     |
| Amazon SES   | `email-smtp.<region>.amazonaws.com` | 587 | SMTP credentials (generated in AWS Console) |
| Mailgun      | `smtp.mailgun.org`       | 587        | `postmaster@yourdomain` + password / API key |
| Postmark     | `smtp.postmarkapp.com`   | 587        | `apikey` + server token  |
| Corporate    | `mail.your-company.com`  | 587 (or 25/465) | LDAP or service account |

If you use a consumer email service (Gmail, Outlook) for testing, you must generate an **app‑specific password** – standard account passwords are blocked for SMTP. Note that consumer services have strict sending limits and are not suitable for production.

### 2.2 Obtain SMTP Credentials

The exact procedure varies by provider. As a representative example, using **SendGrid**:

1. Create a SendGrid account (free tier available).
2. Navigate to **Settings > API Keys**.
3. Click **"Create API Key"**, assign a name (e.g., `AlignmentSMTP`), and grant **"Mail Send"** permissions.
4. Copy the generated API key – it will not be shown again.
5. For SMTP authentication, use the username `apikey` and the copied API key as the password.
6. The `from_addr` should be a verified sender address (domain or email address you have registered with SendGrid).

For **Amazon SES**:

1. Verify a domain or email address in the SES Console.
2. Navigate to **SMTP Settings** and create a set of SMTP credentials (these are distinct from your AWS IAM credentials).
3. Note the SMTP server endpoint for your region (e.g., `email-smtp.us-east-1.amazonaws.com`).
4. Use the generated username and password in the configuration.

For a **corporate SMTP server**:

- Contact your IT department to obtain:
  - Server hostname (e.g., `smtp.internal.company.com`).
  - Port (usually `587` for TLS, `465` for SSL, or `25` for unencrypted).
  - Authentication method (typically `LOGIN` or `PLAIN`) and credentials (service account).
  - Sender email address that is allowed to relay through the server.

### 2.3 Security Considerations

- Never store SMTP credentials in plain text in configuration files.
- Use a dedicated **service account** with the minimum necessary permissions – do not use personal email credentials.
- If possible, enable **TLS/STARTTLS** (port 587) or **SSL** (port 465) to encrypt the connection.
- Rotate credentials periodically and revoke them if a compromise is suspected.
- In cloud environments, consider using the provider's native email sending API (e.g., AWS SES SDK) instead of SMTP to avoid managing passwords.

### 2.4 Configuration in `alignment_config.yaml`

Below is a full example using SendGrid:

```yaml
human_oversight:
  channels:
    email:
      smtp_server: "smtp.sendgrid.net"
      smtp_port: 587
      username: "apikey"
      password: "SG.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
      from_addr: "alignment-alerts@your-verified-domain.com"
```

For a corporate server:

```yaml
human_oversight:
  channels:
    email:
      smtp_server: "mail.your-company.com"
      smtp_port: 587
      username: "svc_alignment"
      password: "secure_password_here"
      from_addr: "ai-alerts@your-company.com"
```

The `from_addr` must be an email address that the SMTP server accepts as a legitimate sender.

### 2.5 Testing the Email Channel

After configuration, you can manually test the email delivery by calling the `HumanOversightInterface.request_intervention` method with a dummy report and `channels=["email"]`. The module will attempt to send an email to a hard‑coded recipient address (`human-reviewer@example.com`). In production, you should extend the `EmailAdapter` to dynamically determine the reviewer's email address, for example by looking up the operator ID in a directory service.

## 3. Verifying Both Channels

Once the configuration is complete, the `HumanOversightInterface` automatically initialises the corresponding channel adapters when the `webhook_url` or `smtp_server` fields contain valid values (i.e., not the placeholder `"..."`). The adapters are only activated if the required configuration is present.

To confirm that a channel is properly set up, examine the logs for lines such as:

```
[INFO] Human Oversight: Slack adapter initialised with webhook
[INFO] Human Oversight: Email adapter initialised with SMTP server smtp.sendgrid.net
```

If a channel fails to initialise (e.g., missing credentials), the system logs a warning and continues without that channel.

## 4. Summary

| Channel | Required Configuration Items | Security Practice |
|---------|------------------------------|-------------------|
| Slack   | `webhook_url` (from API portal) | Store as secret; use per‑channel granularity |
| Email   | `smtp_server`, `smtp_port`, `username`, `password`, `from_addr` | Use service account; enable TLS; rotate credentials |

After applying these settings, the alignment agent can notify human reviewers via both channels simultaneously, with configurable retries and fallback escalation policies as defined in the `human_oversight.escalation` section of the configuration file.