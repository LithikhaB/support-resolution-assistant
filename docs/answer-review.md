# Telecom troubleshooting answer review

Reviewer name: Lithikha B
Review date: 2026-10-04

Please read these five actual application answers and fill the blank scores. They are synthetic development complaints evaluated with live providers. No AI or human quality scores have been entered. This is an owner review; independent reviewer status must be reported accurately.

Score each dimension: **0 = inadequate, 1 = partly adequate, 2 = adequate**.

- Context: Does the answer understand the complaint and previous attempts?
- Relevance: Are its checks and actions appropriate to this issue?
- Grounding: Do the cited sources support the answer, with repairs conditional on findings?
- Clarity: Could a support agent use the answer without struggling to interpret it?

Leave a short note for every 0 or 1. Do not award grounding solely because a citation exists. Compare its source excerpts with the actual step.

## Answer 1 BD03

Case ID: query_BD03_01

**Complaint**

My first invoice after changing plans is higher and contains part-month entries for both plans. Already checked: Compared invoice dates with the plan-change date. Could you check this when you have the details?

**Application answer**

Youâ€™ve noted that the first invoice after your plan change is higher and shows partâ€‘month entries for both plans. The next step is to confirm the ledgerâ€™s prorated periods and explain the line items if they are correct.

1. Support: Verify that the ledger shows correct, nonâ€‘overlapping prorated periods and no duplicate charge for the invoice. [S1]
2. You can check: If the ledger confirms the periods are correct, explain the dated line items and calculation; retain the valid charges unless new evidence shows an error. [S1]

Do not issue a credit merely because the customer is angry. [S1]

**Source excerpts**

[S1] Valid prorated billing misunderstood. â€” procedure
- condition: Ledger shows correct non-overlapping prorated periods and no duplicate charge.
- action: Explain the dated line items and calculation; retain the valid charges unless new evidence shows an error.
- restriction: Do not issue a credit merely because the customer is angry.

[T1] Billing dispute support case (similar_category; simulated outcome)
Finding: Ledger confirms duplicate settled payments against one invoice. Action: Authorized billing staff reconcile the ledger and initiate the approved duplicate-payment adjustment. Simulated follow-up: adjustment appears in the ledger and is communicated to the customer.

[T2] Billing dispute support case (similar_category; simulated outcome)
Finding: Audit trail has no valid consent record for the add-on. Action: Billing staff investigate consent, remove the unsupported charge through policy, and document the decision. Simulated follow-up: corrected invoice excludes the unsupported add-on.

**Your ratings**

| Context 0–2 | Relevance 0–2 | Grounding 0–2 | Clarity 0–2 |
|---|---|---|---|
| 1 | 2 | 2 | 1 |

Comments or corrections: 
Context 1: Does not acknowledge that the invoice dates were already compared with the plan-change date.
Clarity 1: The step 2 label "You can check" is odd, since it is really an explanation to give the customer. The "do not issue a credit merely because the customer is angry" line does not fit a calm complaint.

Acceptable for a demo? Yes


## Answer 2 ID03

Case ID: query_ID03_01

**Complaint**

The internet disconnects for a minute at almost the same interval on Ethernet and Wi-Fi, with no change in the fibre light. Already checked: Restarted the router and recorded three disconnect times. Could you check this when you have the details?

**Application answer**

Youâ€™re experiencing brief, regular disconnects on both wired and wireless connections while the fibre light stays steady. The next step is to verify whether the gateway is having trouble renewing its lease, which can cause these brief outages.

1. You can check: review the gateway logs for any session renewal failures or stale lease records that occur around the times of the disconnects. [S1]
2. Support: If the logs confirm a stale lease issue, the authorized access team will clear the stale lease and apply the validated renewal configuration. [S1]

Do not diagnose a broken optical cable without optical fault evidence. [S1]

**Source excerpts**

[S1] Access-session renewal failure. â€” procedure
- condition: Gateway logs show session renewals failing against a stale lease record.
- action: Authorized access team clears the stale lease and applies the validated renewal configuration.
- restriction: Do not diagnose a broken optical cable without optical fault evidence.

[T1] Intermittent broadband support case (similar_category; simulated outcome)
Finding: Optical loss events align with rain; technician finds moisture at an external joint. Action: Dispatch an authorized technician to repair and seal the damaged joint. Simulated follow-up: no optical loss events during the next wet-weather observation.

[T2] Intermittent broadband support case (similar_category; simulated outcome)
Finding: Line stays synchronized; provider monitoring shows peak-hour congestion on the access segment. Action: Network operations rebalance capacity and monitor the affected segment during peak hours. Simulated follow-up: peak-hour packet loss subsides and calls stay connected.

**Your ratings**

| Context 0–2 | Relevance 0–2 | Grounding 0–2 | Clarity 0–2 |
|---|---|---|---|
| 2 | 2 | 2 | 1 |

Comments or corrections: 
Clarity 1: "You can check: review the gateway logs" assigns an agent-side task to the customer's side. Who does what is unclear.

Acceptable for a demo? Needs correction


## Answer 3 MC03

Case ID: query_MC03_01

**Complaint**

My phone has no service after I manually selected a network on a trip. Other phones here work. Already checked: Restarted once; the manually selected network remains configured. Could you check this when you have the details?

**Application answer**

You can try switching the phone back to automatic network selection to see if service returns. If the device is still locked to an unavailable operator, support will need to confirm and adjust the operator settings.

1. You can check: Go to Settings > Mobile network and set the network selection to automatic. [S1]
2. Support: Verify that the device is not locked to an unavailable operator and, if so, change the operator selection to automatic. [S1]

Do not replace an active SIM before checking operator selection. [S1]

**Source excerpts**

[S1] Manual operator selection left enabled. â€” procedure
- condition: Device is locked to an unavailable operator; account and SIM are active.
- action: Return network selection to automatic using the supported device settings.
- restriction: Do not replace an active SIM before checking operator selection.

[T1] Mobile coverage support case (similar_category; simulated outcome)
Finding: Network incident confirms an affected serving site. Action: Associate the reports with the network incident and provide the confirmed status; operations restore the site. Simulated follow-up: test devices register after site restoration.

[T2] Mobile coverage support case (similar_category; simulated outcome)
Finding: No area outage; indoor attenuation is reproducible on multiple devices. Action: Assess supported Wi-Fi calling and record the location for coverage review. Simulated follow-up: supported Wi-Fi calling works indoors; outdoor coverage remains normal.

**Your ratings**

| Context 0–2 | Relevance 0–2 | Grounding 0–2 | Clarity 0–2 |
|---|---|---|---|
| 2 | 2 | 2 | 2 |

Comments or corrections: 
None. The menu path "Settings > Mobile network" is not in the source, but the source only says "supported device settings", so no score was reduced.

Acceptable for a demo? Yes


## Answer 4 SM03

Case ID: query_SM03_01

**Complaint**

Since changing phones, messages from one contact are missing, but other people can text me. Already checked: Checked the handset's blocked and spam folders. Could you check this when you have the details?

**Application answer**

You can check whether the contact is blocked locally. If the block is confirmed, support can remove it or adjust the filtering setting.

1. You can check: Open your messaging appâ€™s blocked contacts list and look for the missing contactâ€™s number. [S1]
2. Support: If the contact is found in the blocked list, remove the block or adjust the deviceâ€™s filtering setting with your consent. [S1]

Do not disable all spam protection or expose message contents in logs. [S1]

**Source excerpts**

[S1] Local message filtering. â€” procedure
- condition: The sender is blocked locally; network delivery receipts are successful.
- action: With consent, remove the unintended block or adjust the device's supported filtering setting.
- restriction: Do not disable all spam protection or expose message contents in logs.

[T1] Sms otp support case (similar_category; simulated outcome)
Finding: Network trace shows no message submission from the affected sender; other SMS delivers. Action: Provide trace findings and refer the customer to the sender's official support channel. Simulated follow-up: sender fixes its submission and a new verification message arrives.

[T2] Sms otp support case (similar_category; simulated outcome)
Finding: Messaging profile contains an invalid service-center configuration. Action: Authorized support restores the provider-approved messaging configuration. Simulated follow-up: ordinary outgoing test messages deliver.

**Your ratings**

| Context 0–2 | Relevance 0–2 | Grounding 0–2 | Clarity 0–2 |
|---|---|---|---|
| 1 | 1 | 1 | 2 |

Comments or corrections: 
Context 1: The customer already checked the blocked and spam folders on the handset, and the answer sends them back to the blocked list.
Relevance 1: Repeats an attempted step and does not address what changed with the phone switch.
Grounding 1: The source condition also requires successful network delivery receipts, and the answer never asks to verify that. The block removal is conditional, which is good.

Acceptable for a demo? Needs correction


## Answer 5 VC03

Case ID: query_VC03_01

**Complaint**

My new compatible phone has fast data but calls fail immediately. The same SIM could call in my old phone. Already checked: Confirmed device compatibility and restarted once. Could you check this when you have the details?

**Application answer**

Youâ€™re experiencing call failures on a new phone while data works fine, and the same SIM works on an older device. The next step is to verify whether Voiceâ€‘overâ€‘LTE registration is missing for your line. [T1] [T2] Historical references are simulated examples.

1. Support: Verify that Voiceâ€‘overâ€‘LTE registration is missing for the line despite compatible hardware. [S1]
2. If confirmed, Support: Authorized provisioning team restores the supported voice service profile. [S1]

Do not instruct the customer to place emergency calls as a test. [S1]

**Source excerpts**

[S1] IMS voice provisioning missing. â€” procedure
- condition: Voice-over-LTE registration is missing for the line despite compatible hardware.
- action: Authorized provisioning team restores the supported voice service profile.
- restriction: Do not instruct the customer to place emergency calls as a test.

[T1] Voice call failure support case (similar_category; simulated outcome)
Finding: Account has an unintended outgoing-call restriction; voice registration is otherwise healthy. Action: Authorized account staff verify identity and remove the unintended restriction. Simulated follow-up: outgoing test calls to two normal numbers connect.

[T2] Voice call failure support case (similar_category; simulated outcome)
Finding: Network traces show a voice handover failure between serving cells. Action: Radio operations correct the affected handover configuration and retest. Simulated follow-up: calls continue through the previously failing route.

**Your ratings**

| Context 0–2 | Relevance 0–2 | Grounding 0–2 | Clarity 0–2 |
|---|---|---|---|
| 2 | 2 | 1 | 1 |

Comments or corrections: 
Grounding 1: The intro cites [T1] and [T2] for the VoLTE hypothesis, but those cases are a call restriction and a handover failure, so they do not support it. S1 itself is used correctly.
Clarity 1: The line "[T1] [T2] Historical references are simulated examples" is confusing for an agent and adds nothing to the actionable steps.

Acceptable for a demo? Needs correction

## Return the review

Send the four scores and notes for each case ID in chat, or edit this document and save it. Missing ratings will remain missing. These one-turn examples do not evaluate follow-up conversation quality.

Evidence: `data/evaluation/live_review_partial_20261004.json`. The live evaluation is still in progress; this document freezes the displayed answers for your review.
