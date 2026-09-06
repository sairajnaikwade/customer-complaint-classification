# Customer Complaint Dataset Documentation

## 1. Overview
This dataset is developed for the **Customer Complaint Classification System** under the NLP PBL curriculum at **Sanjivani College of Engineering, Kopargaon** (Academic Year 2026-27). It contains real-world customer complaints across 6 predefined domain categories designed for supervised text classification.

## 2. Dataset Summary
- **Total Records:** 1,200 labeled complaints
- **Features:** 2 (`complaint_text`, `category`)
- **Balance:** Perfectly balanced (200 records per class)
- **Format:** Comma-Separated Values (`.csv`)
- **Encoding:** UTF-8

## 3. Categories & Target Distribution
| Category | Samples | Target Department | Description & Key Patterns |
| :--- | :--- | :--- | :--- |
| **Payment Issue** | 200 | Finance / Payments Team | Failed transactions, UPI/wallet debits without order confirmation, pending refunds. |
| **Billing Issue** | 200 | Billing Team | Duplicate charges, unexpected fees, erroneous invoice totals, auto-renewal issues. |
| **Technical Issue** | 200 | Technical Support | Application crashes, 500 errors, UI freezes, unresponsive buttons, browser glitches. |
| **Delivery Issue** | 200 | Logistics / Delivery Team | Late delivery, wrong address dispatch, damaged parcels, lost shipments. |
| **Account Issue** | 200 | Account Support | Login failures, missing OTP, password reset issues, 2FA failures, account lockouts. |
| **Service Issue** | 200 | Customer Support | Rude representatives, unhelpful agents, delayed ticket response, poor after-sales. |

## 4. Preprocessing Considerations
- Standardized text representations.
- Handled edge cases (empty strings, special symbols, alphanumeric tokens).
- Maintained domain-relevant terminology while discarding redundant conversational stops.
