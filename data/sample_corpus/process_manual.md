# NorthArc Process Manual — Onboarding & Incident Response

## New-Hire Onboarding

Every new hire is paired with an **onboarding buddy** for their first 30 days. The
buddy is not the new hire's manager and is responsible for answering day-to-day
questions, introducing the new hire to relevant working groups, and flagging any
blockers to the manager.

### Week 1 checklist

1. Laptop pickup at the IT desk (open 09:00–17:00, Mon–Fri).
2. Complete the **Security & Compliance 101** course in the Learning Portal — this
   is mandatory before any client system access is granted.
3. 1:1 with the manager (60 minutes) to review the first-30-day plan.
4. Coffee chat with the onboarding buddy.
5. Read the firm handbook and acknowledge receipt in Workday.

### Weeks 2–4

- Shadow at least one client meeting.
- Complete the **Practice Area Primer** for your assigned vertical.
- Submit your first internal time entry by end of Week 2.
- A 30-day check-in with HR is scheduled automatically.

Probation is **90 days**. Confirmation of employment requires sign-off from the
manager and the practice lead.

## Incident Response (Production)

NorthArc operates a Sev1–Sev4 model for production incidents involving client
systems or the Insight platform.

| Severity | Definition | Response time | Notification |
|---|---|---|---|
| Sev1 | Total outage or data-loss event | 15 minutes | CTO + on-call CSM |
| Sev2 | Major feature degraded for >25% of users | 30 minutes | Practice lead |
| Sev3 | Minor feature degraded; workaround exists | 4 business hours | Team channel |
| Sev4 | Cosmetic or low-impact | next business day | Ticket queue |

### On-call rotation

Engineers rotate weekly. The on-call engineer is **PagerDuty-paged** for Sev1 and
Sev2. The handoff happens every Monday at **10:00 local time** via a 15-minute
sync in the `#oncall` channel.

### Post-incident review

Every Sev1 and Sev2 requires a written **blameless post-mortem** published within
**five business days**. The post-mortem must include: timeline, contributing
factors, customer impact, what worked, what did not, and at least one durable
action item with an owner and due date.

## Travel Approval

Domestic travel under $1,500 requires only manager approval through Concur.
International travel, or any trip exceeding **$2,500**, additionally requires
practice-lead approval and must be booked at least **14 days** in advance to
qualify for reimbursement. Last-minute bookings require a written justification.
