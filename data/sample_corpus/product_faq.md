# NorthArc Insight Platform — Customer FAQ

## What is Insight?

Insight is NorthArc's hosted analytics workbench. It ingests structured and
semi-structured client data, runs scheduled transformations, and surfaces results
through dashboards and an embedded notebook environment. Insight is sold under three
plans: **Starter**, **Team**, and **Enterprise**.

## Pricing

| Plan | Monthly price | Users | Data volume | Support |
|---|---|---|---|---|
| Starter | $499 | up to 5 | up to 50 GB | community + email |
| Team | $1,499 | up to 25 | up to 500 GB | business hours |
| Enterprise | custom | unlimited | custom | 24×7, named CSM |

Annual contracts receive a **15% discount** on the equivalent monthly price.
Mid-contract upgrades are pro-rated; downgrades take effect at the next renewal.

## Supported data sources

Out of the box, Insight connects to: PostgreSQL, MySQL, Snowflake, BigQuery,
Redshift, Salesforce, HubSpot, Stripe, Google Sheets, S3 (CSV/Parquet/JSON), and
Azure Blob Storage. Custom connectors can be added via our Python SDK; a Java SDK
is on the **Q3 2026 roadmap** and is not yet available.

## Data residency

Customers may pick a primary region at provisioning time:
**us-east-1**, **eu-west-1**, or **ap-south-1**. Data does not leave the chosen
region. EU customers on the Enterprise plan may additionally enable the
**EU Data Boundary** add-on, which keeps logs and metadata in-region as well.

## Security

Insight is **SOC 2 Type II** certified and ISO 27001 compliant. All data at rest is
encrypted with AES-256. Data in transit uses TLS 1.3. SSO is supported via SAML 2.0
(Okta, Azure AD, OneLogin) on Team and Enterprise plans.

## Service-Level Agreement

Enterprise customers receive a **99.95% monthly uptime** SLA, with service credits
of 10% for any month below 99.9% and 25% below 99.0%. Team and Starter plans are
provided on a best-effort basis with no formal SLA.

## Cancellation

Monthly customers can cancel any time; service continues to the end of the paid
period. Annual customers may cancel mid-term but are not refunded for the unused
portion. Data export tools remain accessible for **30 days** after cancellation,
after which the workspace is permanently deleted.
