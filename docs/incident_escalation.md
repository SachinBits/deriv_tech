# Incident Severity and Escalation

This runbook tells support staff how to classify production incidents and who to involve.

## Severity levels

P1 means a full outage or a problem that blocks most customers. P2 means a major feature is degraded for many customers. P3 means a minor issue with a workaround available.

## Response targets

For P1 incidents, the on-call engineer must respond within 15 minutes. For P2 incidents, the on-call engineer must respond within 1 hour. For P3 incidents, the on-call engineer must respond within 1 business day.

## Escalation path

Incidents escalate from support to the on-call SRE, and from the on-call SRE to the engineering manager. Support opens the incident ticket, records the severity and pages the on-call SRE. If the on-call SRE cannot stabilise the problem, they escalate to the engineering manager.

## Communication

Support posts status updates in the incident channel at regular intervals until the incident is resolved. After resolution, the on-call SRE writes a short post-incident summary describing the cause and the fix.
