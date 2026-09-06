"""
Dataset Generator for Customer Complaint Classification System
College NLP PBL Project - Academic Year 2026-27
Sanjivani College of Engineering, Kopargaon

Generates a balanced, realistic, multi-class customer complaint dataset
covering 6 predefined categories with rich vocabulary and realistic linguistic variations.
"""

import os
import pandas as pd
import random

# Fix random seed for reproducibility
random.seed(42)

CATEGORIES = [
    "Payment Issue",
    "Billing Issue",
    "Technical Issue",
    "Delivery Issue",
    "Account Issue",
    "Service Issue"
]

TEMPLATES = {
    "Payment Issue": [
        "My payment was deducted but the order was not confirmed.",
        "Money was debited from my bank account via UPI but transaction shows failed.",
        "I tried paying using Google Pay, amount deducted but payment status is still pending.",
        "My credit card was charged for transaction {id}, but no order receipt was generated.",
        "Payment failed at the gateway checkout but amount was cut from my savings account.",
        "I requested a refund 7 days ago for a failed payment, still haven't received my money back.",
        "Net banking transaction timed out during payment verification step.",
        "My Paytm wallet was debited twice for a single order purchase.",
        "Payment gateway error: transaction was debited but vendor claims payment is unreceived.",
        "Refund for cancelled order #{id} is not credited to my original payment method.",
        "The money was deducted via PhonePe but the cart emptied without confirmation.",
        "Debit card OTP was submitted successfully, money deducted, yet payment marked as failed.",
        "Payment was processed twice because the gateway page froze on the first attempt.",
        "I paid through debit card but received an SMS saying payment declined after deduction.",
        "My transaction failed at checkout due to payment gateway error #{id}.",
        "Bank statement shows deduction of Rs {amount} but order history is empty.",
        "Initiated refund for transaction #{id} is still pending after two weeks.",
        "Payment processing failed during merchant authorization but money left my account.",
        "I made a payment via Apple Pay and the money was deducted without an order confirmation.",
        "The amount Rs {amount} was charged to my card but the transaction status shows incomplete.",
        "Double deduction occurred on my bank account for single transaction #{id}.",
        "Checkout page showed transaction error but bank sent deduction alert immediately.",
        "Payment gateway keeps declining my card even though there are sufficient funds.",
        "Cashback and refund amount of Rs {amount} not credited back to my account.",
        "Auto-debit triggered on my bank account for an order that was cancelled.",
        "I did not receive any acknowledgement or receipt for my successful UPI payment.",
        "Money deducted from card but website displayed error code PAY_{id}.",
        "Payment transfer failed halfway and my funds are currently stuck in escrow.",
        "Refund amount processed by seller is not reflecting in my linked bank account.",
        "Credit card charge went through successfully but order was marked as unpaid.",
        "Transaction ID {id} shows success on bank portal but failed on the shopping site.",
        "I paid the full amount but only half of the order was processed by the payment system.",
        "My UPI transfer to merchant {id} got debited but the merchant says payment not received.",
        "The refund initiated 10 days back hasn't appeared in my bank statement yet.",
        "My saved card was auto-charged for a subscription I cancelled 2 months ago.",
        "Amount Rs {amount} is deducted from account but checkout session expired.",
        "I am unable to complete payment because the OTP keeps failing on every attempt.",
        "The PayPal transaction {id} was processed but the order shows pending payment.",
        "Rupay card payment declined repeatedly even after bank confirmed no issues.",
        "I made an IMPS transfer but the beneficiary account didn't receive the money.",
        "International card transaction was blocked but the amount was still reserved.",
        "Wallet balance deducted but transaction failed at the last step of checkout.",
        "NACH mandate deducted EMI from my account but the loan account isn't updated.",
        "Card payment gateway timeout caused double charging on the same invoice.",
        "I cannot initiate a refund because the refund option is disabled for my order.",
        "My reward points were debited but the payment still charged full amount.",
        "Payme transfer of Rs {amount} shows pending on both sender and receiver ends.",
        "The payment failed message appeared but my bank balance reduced significantly.",
        "I haven't received confirmation for the insurance premium payment I made online.",
        "Bank shows payment successful but the utility company says dues are still pending.",
    ],
    "Billing Issue": [
        "My card was charged twice for the exact same order invoice.",
        "I noticed an unexpected hidden fee of Rs {amount} on my latest monthly bill.",
        "My monthly subscription auto-renewed and billed my account without any prior reminder.",
        "The invoice shows an incorrect total amount that does not match the cart price.",
        "Discount promo code was applied at checkout but invoice billed full price without discount.",
        "I was billed for an additional service that I never opted in or subscribed for.",
        "Double billing on my monthly recurring statement for invoice #{id}.",
        "Tax calculation on my invoice is erroneous and exceeds the standard GST percentage.",
        "I cancelled my premium membership last month but was still billed for this billing cycle.",
        "The price listed on the invoice #{id} is higher than the product listing price.",
        "I was charged an extra convenience fee of Rs {amount} that was not mentioned before.",
        "My billing receipt has incorrect company details and wrong GST number.",
        "Incorrect surge pricing added to my final invoice without notification.",
        "I received two conflicting invoices for a single subscription plan.",
        "Billed for delivery charges even though my order qualified for free shipping.",
        "Invoice #{id} contains items that were out of stock and removed from my order.",
        "Recurring credit card charge occurred after I explicitly cancelled the renewal.",
        "My billing statement shows charges from your company that I did not authorize.",
        "Discrepancy in billing: invoice total is Rs {amount} while order total was different.",
        "Cancellation fee was wrongly charged to my account after cancelling within free window.",
        "I was billed twice for annual maintenance fee on invoice #{id}.",
        "The coupon code discount of 20% was omitted from my final generated invoice.",
        "Duplicate invoice generated for account #{id} resulting in duplicate bank charges.",
        "I was charged an unexplained fee labelled miscellaneous charges on my statement.",
        "Subscription downgrade was requested but I am still being billed the enterprise rate.",
        "Receipt shows wrong currency conversion rate leading to overcharge on my card.",
        "I have been billed for equipment rental even though I returned the device.",
        "Billing department did not adjust the credit note on my current statement.",
        "My invoice displays an outdated billing address despite updating profile details.",
        "Extra charge of Rs {amount} added for expedited delivery that arrived late.",
        "I was not informed about the service charge increase before my renewal date.",
        "The tax invoice shows CGST and SGST incorrectly calculated for my state.",
        "Old tariff plan still being applied despite upgrading to new plan #{id}.",
        "My employer reimbursement invoice is missing the required company GST number.",
        "Late payment fee charged even though I paid 5 days before the due date.",
        "I was charged the international rate even though I was browsing from India.",
        "The pro-rated billing amount for partial month is incorrectly calculated.",
        "Received invoice #{id} with wrong product description and wrong unit price.",
        "I opted for a quarterly billing cycle but was charged monthly with higher rates.",
        "Security deposit of Rs {amount} was not reflected as credit in the final bill.",
        "Incorrect penalty fee added for data overage that did not actually occur.",
        "Two separate bills generated for a single service period causing confusion.",
        "The early termination fee on my account exceeds the contractual limit.",
        "I was charged for both the old and new plan during the same billing period.",
        "Platform fee deducted from my seller payout without prior disclosure.",
        "Annual plan was purchased but monthly charges continue to appear on statement.",
        "My billing cycle date changed without notification causing mid-month deduction.",
        "GST invoice not generated despite repeated requests to the billing department.",
        "The discount applied at sales time is not reflected in the final tax invoice.",
        "My corporate billing account is receiving invoices for a personal account charges.",
    ],
    "Technical Issue": [
        "The application keeps crashing whenever I try to open the dashboard.",
        "The mobile app crashes immediately after I enter my login credentials.",
        "Website displays internal server error 500 when navigating to the checkout screen.",
        "Web page freezes completely when attempting to upload documents.",
        "Search bar on the web application is broken and returns no results for any query.",
        "I am receiving a persistent JavaScript console error when clicking the submit button.",
        "The app is stuck on the infinite loading spinner on the home screen.",
        "Cannot download my generated PDF report due to server timeout error #{id}.",
        "UI buttons are unresponsive and misaligned on mobile screen resolution.",
        "The application crashes on iOS version 17 when accessing camera permissions.",
        "Page fails to render correctly on Google Chrome browser with blank white screen.",
        "API endpoint /api/v1/data is throwing HTTP 502 Bad Gateway intermittently.",
        "Filter dropdown is not functioning and freezes the browser tab.",
        "The app crashes with an uncaught null pointer exception when editing profile.",
        "Data synchronization between mobile app and web portal is failing constantly.",
        "Notification badge is stuck showing 99+ unread messages that cannot be cleared.",
        "Image upload fails with error code 413 payload too large for small photos.",
        "The application restarts automatically every time I navigate to the settings tab.",
        "Session expires abruptly within 30 seconds of logging into the portal.",
        "The checkout button is greyed out and unclickable despite filling all fields.",
        "Dark mode toggle breaks the layout and causes overlapping text elements.",
        "Encountering database connection error during search query execution.",
        "App performance is extremely sluggish with severe latency during page transitions.",
        "The application crashes whenever a push notification arrives.",
        "Form validation error appears even when all mandatory input fields are correctly filled.",
        "Voice input feature fails to initialize audio stream on Android app.",
        "The web dashboard does not load data charts and shows an empty canvas.",
        "QR code scanner within the app opens a black screen and crashes.",
        "Export to Excel button throws error 503 Service Unavailable.",
        "The interface suffers from severe memory leaks causing browser tab to crash.",
        "The app freezes on the splash screen and never loads past the logo animation.",
        "Video call feature drops connection every two minutes with codec error.",
        "Print invoice button opens empty dialog box with no content inside.",
        "Authentication token refresh is failing causing automatic logouts.",
        "The map integration on the delivery tracking page shows a blank tile.",
        "File attachment in the support ticket form fails with unknown mime error.",
        "App crashes when switching between portrait and landscape orientation.",
        "Real-time chat messages are not being delivered instantly.",
        "The PWA does not install correctly on Android Chrome browser.",
        "Barcode scanner freezes the device camera when scanning product codes.",
        "Two-way sync between desktop and mobile keeps reverting changes.",
        "The background location service crashes the app on battery saver mode.",
        "Push notifications are not appearing even with all permissions granted.",
        "CSS layout breaks completely when the browser zoom exceeds 110 percent.",
        "Calendar widget fails to load available slots showing empty months.",
        "Auto-complete suggestion dropdown appears behind other page elements.",
        "App crashes on startup after the latest update was installed.",
        "Screen recording feature produces corrupted video output files.",
        "Data export function generates empty CSV files without content.",
        "The map navigation feature shows roads that no longer exist.",
    ],
    "Delivery Issue": [
        "My order has not arrived yet even though the estimated delivery date was yesterday.",
        "My package was marked as delivered on the tracker but I have not received it.",
        "The courier agent delivered my parcel to the wrong address in another street.",
        "The package arrived severely damaged with broken contents inside the box.",
        "Items were missing from my shipment box when I opened the sealed parcel.",
        "Tracking status has been stuck at 'In Transit' for over 10 consecutive days.",
        "Courier delivery executive claimed 'Customer not available' without attempting delivery or calling.",
        "Delivery agent was rude and refused to deliver the parcel to my doorstep.",
        "Package #{id} was returned to sender without my consent or any delivery attempt.",
        "Received the completely wrong product instead of the item I originally ordered.",
        "Estimated delivery date has been rescheduled 4 times without any explanation.",
        "Parcel was left unattended outside my gate in the rain and got ruined.",
        "Delivery executive demanded extra cash on delivery amount beyond invoice total.",
        "Express shipping paid for next-day delivery took over a week to arrive.",
        "Order shipment #{id} is lost in transit and logistics support has no update.",
        "Tampered packaging received with broken seal and missing electronic items.",
        "Courier company marked fake delivery attempt stating wrong address provided.",
        "My order status says out for delivery for the past 48 hours with no courier contact.",
        "The delivery partner refused to allow open box inspection as promised.",
        "Shipment was dispatched to the wrong distribution hub in a different state.",
        "Package has been sitting at the local courier facility for 5 days without dispatch.",
        "Delivery boy demanded delivery charges despite pre-paid order status.",
        "My parcel tracking ID is showing invalid on the courier tracking website.",
        "I received an empty box with packing material and no ordered goods inside.",
        "Delivery was scheduled for morning slot but arrived at 11 PM at night.",
        "Order delivered to an unknown neighbor without contacting me for an OTP.",
        "Logistics tracking shows delivered with a forged signature that is not mine.",
        "Perishable goods arrived spoiled due to extensive courier transit delays.",
        "Delivery address was updated before dispatch yet shipment was sent to old address.",
        "Courier service returned package citing address untraceable despite clear landmarks.",
        "The AWB number for my shipment is not trackable on any courier website.",
        "My international parcel is stuck in customs for 3 weeks with no update.",
        "Delivery partner attempted delivery on a public holiday without prior notice.",
        "The heavy furniture item was left in the lobby and not assembled as promised.",
        "My cold chain shipment arrived at room temperature and is unsafe to consume.",
        "The replacement phone arrived with cosmetic damage from poor packaging.",
        "I was not given any delivery OTP prompt and the item was handed to someone else.",
        "My subscription box for this month was never dispatched despite being charged.",
        "The courier broke a fragile electronic item by placing heavy boxes on top.",
        "My drone delivery dropped the package in the wrong GPS coordinates location.",
        "Two out of four ordered items were missing from an otherwise sealed carton.",
        "The hyperlocal delivery promised in 2 hours took more than 18 hours to arrive.",
        "My parcel shows custom clearance failed but no reason or action step is given.",
        "The return pickup was scheduled but courier never arrived for collection.",
        "My address was on the package label but the shipment went to another city entirely.",
        "The delivery slot I selected was ignored and delivery attempted at wrong time.",
        "The oversize item was refused delivery by the apartment security with no resolution.",
        "My urgent medical supplies delivery was delayed by 72 hours without notification.",
        "The tracking link shared via SMS leads to a broken 404 page.",
        "Wrong pin code was printed on the label despite correct address in order details.",
    ],
    "Account Issue": [
        "I cannot login to my account and password reset link is not being sent.",
        "OTP verification code is not being received on my registered mobile number.",
        "My account has been temporarily locked due to suspicious activity without warning.",
        "I forgot my account password and cannot access the registered recovery email.",
        "Two-factor authentication 2FA code generated by authenticator app is rejected as invalid.",
        "Unable to update my primary phone number and email address in account profile.",
        "I received a notification of an unauthorized login to my account from an unknown IP.",
        "My account was suspended without any reason or violation notice.",
        "Cannot deactivate or delete my old account despite following instructions.",
        "KYC document verification is stuck in pending review status for two weeks.",
        "Unable to merge my duplicate user profiles registered under the same email.",
        "Login screen says account does not exist even though I registered yesterday.",
        "Cannot link my Google OAuth or Social Login account with existing profile.",
        "Biometric fingerprint login fails repeatedly and prompts for password reset.",
        "Security questions validation fails despite entering correct answers.",
        "I am unable to change my account username and security credentials.",
        "Account session keeps logging me out immediately after successful authentication.",
        "Registered email address has a typo and I cannot verify my new user account.",
        "User account permissions were unexpectedly revoked by the system.",
        "My account profile data and saved preferences vanished after recent update.",
        "Cannot complete mandatory two-step verification because SMS gateway is failing.",
        "Received phishing warning alert regarding compromised password on my account.",
        "Account recovery team has not responded to my identity verification ticket #{id}.",
        "Captcha on the user login page is constantly showing verification failed error.",
        "Unable to switch between multiple business accounts linked to my master ID.",
        "Child account access cannot be authorized from parent account dashboard.",
        "Password reset token expired error shown immediately upon clicking reset link.",
        "My account has been blocked after 3 incorrect password attempts with no unlock option.",
        "Account verification email is going to spam folder and verification link is broken.",
        "Cannot unlink old mobile number from my registered user profile.",
        "The profile picture upload fails every time I try to change my avatar.",
        "My account shows an incorrect subscription tier despite paying for premium.",
        "Email change request was submitted but confirmation not sent to new address.",
        "Two accounts exist for the same PAN number causing conflicts in the system.",
        "My minor child account was auto-upgraded to adult without parental consent.",
        "Dark pattern in settings made me accidentally close my account.",
        "All saved addresses in my account have disappeared after the app update.",
        "My wishlist and cart were emptied without any action from my side.",
        "The account dashboard is showing another user's personal information.",
        "My loyalty points balance reset to zero after account migration.",
        "Phone number used for account verification is no longer accessible to me.",
        "My account was merged with a different user account without my permission.",
        "I cannot add a second email address to my account as backup contact.",
        "The account age verification keeps failing despite submitting valid documents.",
        "My organization admin removed my access but I should still have permissions.",
        "Profile section crashes whenever I attempt to save updated contact details.",
        "My linked payment method was removed from the account without my authorization.",
        "The account shows logged in from multiple devices I don't own.",
        "I changed my registered email but still receiving OTP on the old email address.",
        "Re-registration was attempted but system says account already exists with same info.",
    ],
    "Service Issue": [
        "Customer support is not responding to any of my emails or support tickets.",
        "The support representative was extremely rude and unhelpful during our phone call.",
        "My grievance ticket #{id} was marked as resolved without any solution provided.",
        "Waited on customer helpline on hold for over 45 minutes and the call got disconnected.",
        "Customer service representative provided inaccurate and misleading information.",
        "No response from support team regarding my critical issue escalated last week.",
        "The chat support bot is useless and loops around without connecting to a human agent.",
        "I requested a manager callback three days ago but have received no follow-up.",
        "Customer relationship executive promised a resolution within 24 hours but failed.",
        "Support agent abruptly disconnected the live chat session while I was typing.",
        "Incompetent support staff unable to comprehend basic product troubleshooting questions.",
        "Terrible after-sales customer service with zero accountability for product defects.",
        "Support portal rejects ticket creation and displays contact administrator message.",
        "Repeated follow-ups on complaint #{id} have been met with canned automated replies.",
        "The service center technician did not show up for the scheduled home appointment.",
        "Customer care hotline numbers are permanently busy or out of service.",
        "Executive refused to escalate my grievance to higher tier management.",
        "Unsatisfactory response provided for warranty claim service request #{id}.",
        "Staff at the local service center was unprofessional and refused service.",
        "Customer support agent closed the live dispute without customer consent.",
        "Promised compensation or callback was never fulfilled by customer relations.",
        "Service quality has deteriorated significantly with unhelpful automated responses.",
        "Helpdesk ticketing system keeps closing tickets automatically without resolution.",
        "Customer care email address is bouncing back with mailbox quota exceeded.",
        "Representative spoke in a dismissive tone and refused to log formal complaint.",
        "Support executive gave conflicting instructions causing further confusion.",
        "No acknowledgment email or ticket number sent after submitting help form.",
        "Service engineer arrived 4 hours late without prior notification.",
        "Customer care supervisor refused to provide refund confirmation in writing.",
        "Poor customer assistance with no resolution provided after weeks of follow-up.",
        "The support team promised a callback within 2 hours but called after 3 days.",
        "Live agent keeps transferring my call between departments without resolution.",
        "The social media support team takes 5 days to reply to direct messages.",
        "The automated IVR system disconnects my call after selecting the correct option.",
        "Support staff denied acknowledging the manufacturing defect in my product.",
        "I was placed in a feedback survey loop instead of being connected to support.",
        "The service warranty was rejected citing reasons not mentioned in the warranty card.",
        "Evening shift customer support provides completely different answers than morning shift.",
        "My formal complaint was downgraded to a general inquiry without my consent.",
        "The support team keeps asking for the same information repeatedly with no progress.",
        "Remote diagnostic tool used by support staff caused data loss on my device.",
        "The onsite technician visit was charged despite the product being under warranty.",
        "Customer service denied my claim citing a clause not present in my agreement.",
        "I was given a replacement that is a lower model than my original defective product.",
        "Support agent was rude when I asked them to transfer me to a senior executive.",
        "The loyalty program points deduction issue I raised was dismissed without investigation.",
        "The feedback submitted after service interaction is never acknowledged or acted upon.",
        "My written grievance letter sent via post was returned as undeliverable.",
        "Service team failed to fix the same issue after 3 separate visits.",
        "The complaint escalation matrix shared on website is completely non-functional.",
    ]
}

# Vocabulary variations and modifiers to generate diverse, realistic complaint instances
VARIATIONS = [
    "Please help resolve this immediately.",
    "This is unacceptable service and needs urgent escalation.",
    "Kindly look into this issue at the earliest.",
    "I have been facing this problem since yesterday.",
    "Very frustrating experience with this platform.",
    "I need an immediate resolution or refund.",
    "Please look into this as soon as possible.",
    "This has happened multiple times this week.",
    "I am extremely disappointed with the handling of this matter.",
    "Urgent attention required for this critical issue.",
    "Kindly investigate and provide an update today.",
    "Looking forward to a quick resolution from your side.",
    "I have already sent multiple emails without response.",
    "This is impacting my daily workflow severely.",
    "Please fix this bug/issue as soon as possible.",
    "Awaiting your prompt reply and action.",
    "I expect a prompt resolution from the concerned department.",
    "Please take necessary action to fix this.",
    "No one is addressing this despite multiple attempts.",
    "Kindly escalate this to senior management."
]


# Cross-category ambiguous complaints (genuine hard examples)
# These deliberately span multiple categories, causing natural model confusion
AMBIGUOUS_COMPLAINTS = [
    ("I cannot access my account to check my payment status.", "Account Issue"),
    ("The app crashed when I was trying to complete my payment.", "Technical Issue"),
    ("My delivery was charged incorrectly on the final invoice.", "Billing Issue"),
    ("Support did not respond even after I raised a payment dispute ticket.", "Service Issue"),
    ("I cannot login to track my package delivery status.", "Account Issue"),
    ("The billing page crashes every time I try to view my invoice.", "Technical Issue"),
    ("Customer support refused to help me with my wrong delivery.", "Service Issue"),
    ("My payment was stuck and I could not reach customer service.", "Payment Issue"),
    ("I paid for expedited delivery but the package never arrived.", "Delivery Issue"),
    ("My account was locked right after I made a payment.", "Account Issue"),
    ("App crashes whenever I check my billing history.", "Technical Issue"),
    ("Wrong item was delivered and support is not responding.", "Service Issue"),
    ("I was double charged and cannot login to request a refund.", "Billing Issue"),
    ("Package shows delivered but my account shows order pending.", "Delivery Issue"),
    ("I cannot update billing address because my account is locked.", "Account Issue"),
    ("The invoice sent by support is different from the one in the app.", "Billing Issue"),
    ("Delivery tracking page throws a server error when I load it.", "Technical Issue"),
    ("Agent promised refund but amount not credited to my payment account.", "Payment Issue"),
    ("My order was cancelled automatically and I was still billed.", "Billing Issue"),
    ("Support told me to check my account but I cannot log in to do so.", "Account Issue"),
    ("Tracking app shows wrong location and customer care is unreachable.", "Delivery Issue"),
    ("The payment gateway fails specifically on the billing summary page.", "Technical Issue"),
    ("I raised a complaint but it was closed without resolving my refund.", "Service Issue"),
    ("Money was debited but delivery has not been initiated yet.", "Payment Issue"),
]

# Common word substitutions for mild text perturbation (simulate user writing styles)
WORD_SYNONYMS = {
    "payment": ["transaction", "payment", "transfer", "deduction"],
    "charged": ["billed", "charged", "debited", "deducted"],
    "account": ["account", "profile", "user account"],
    "login": ["log in", "sign in", "access my account", "login"],
    "delivery": ["shipment", "parcel", "package", "order delivery"],
    "crash": ["crash", "freeze", "stop working", "become unresponsive"],
    "refund": ["refund", "reimbursement", "money back", "credit"],
    "support": ["support", "customer service", "helpdesk", "customer care"],
    "invoice": ["invoice", "bill", "receipt", "billing statement"],
}


def _apply_synonym_perturbation(text: str) -> str:
    """Randomly replaces a few domain words with synonyms to diversify text."""
    words = text.split()
    result = []
    for word in words:
        lower = word.lower().rstrip(".,!?")
        if lower in WORD_SYNONYMS and random.random() < 0.3:
            result.append(random.choice(WORD_SYNONYMS[lower]))
        else:
            result.append(word)
    return " ".join(result)


def generate_complaints_dataset(target_per_category: int = 200) -> pd.DataFrame:
    rows = []
    
    for category, templates in TEMPLATES.items():
        generated_for_category = set()
        count = 0
        
        while count < target_per_category:
            template = random.choice(templates)
            amount = random.choice([150, 499, 999, 1250, 2499, 3500, 4999, 7500, 10000, 15499])
            amount2 = amount + random.choice([50, 100, 250, 500, 1000])
            cid = random.randint(10000, 99999)
            
            text = (template
                    .replace("{id}", str(cid))
                    .replace("{amount}", str(amount))
                    .replace("{amount2}", str(amount2)))
            
            # Synonym-level perturbation (adds diversity without changing label)
            if random.random() > 0.5:
                text = _apply_synonym_perturbation(text)
            
            # Append variation prefix/suffix (increases linguistic diversity)
            if random.random() > 0.4:
                var = random.choice(VARIATIONS)
                text = f"{text} {var}"
            
            text = " ".join(text.split())
            
            if text not in generated_for_category:
                generated_for_category.add(text)
                rows.append({"complaint_text": text, "category": category})
                count += 1

    # Add the ambiguous cross-category complaints
    for text, category in AMBIGUOUS_COMPLAINTS:
        rows.append({"complaint_text": text, "category": category})
    
    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_complaints_dataset(target_per_category=200)
    output_path = os.path.join("dataset", "complaints.csv")
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} complaint samples across {df['category'].nunique()} categories.")
    print("Class distribution:")
    print(df['category'].value_counts())
    print(f"Saved successfully to {output_path}")

