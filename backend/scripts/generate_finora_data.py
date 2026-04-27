"""
FINORA Synthetic Data Generator
================================
Generates all 6 tables needed for the 4 agents:

Agent 1 - Regulatory Watch:
  → internal_policies.csv

Agent 2 - Transaction Risk:
  → user_profiles.csv
  → transactions.csv  (~10,000 rows, ~400 anomalies seeded)

Agent 3 - Market Sentiment:
  → clients.csv
  → sector_news_mapping.json

Agent 4 - Audit Trail:
  → audit_logs.csv  (empty schema, populated at runtime)

Run:
  python3 generate_finora_data.py

Output:
  /data/  folder with all files
"""

import random
import json
import csv
import os
import uuid
import math
from datetime import datetime, timedelta

random.seed(42)  # Reproducible output

OUTPUT_DIR = "./data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def rand_date(start_year=2020, end_year=2024):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    return (start + timedelta(days=random.randint(0, delta.days))).strftime("%Y-%m-%d")

def rand_date_recent():
    """Last 6 months"""
    end = datetime(2024, 12, 31)
    start = datetime(2024, 6, 1)
    delta = end - start
    return (start + timedelta(days=random.randint(0, delta.days))).strftime("%Y-%m-%d")

def rand_datetime(start_year=2024):
    start = datetime(start_year, 1, 1)
    end = datetime(2024, 12, 31)
    delta = end - start
    dt = start + timedelta(
        days=random.randint(0, delta.days),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59)
    )
    return dt.strftime("%Y-%m-%d %H:%M:%S")

INDIAN_FIRST_NAMES = [
    "Aarav", "Arjun", "Rohan", "Vikram", "Rahul", "Amit", "Priya", "Sneha",
    "Anjali", "Kavya", "Neha", "Deepika", "Kiran", "Suresh", "Rajesh",
    "Pooja", "Meera", "Aditya", "Siddharth", "Ishaan", "Riya", "Ananya",
    "Divya", "Shruti", "Nisha", "Tushar", "Varun", "Akash", "Gaurav", "Vivek",
    "Harshit", "Manish", "Nitesh", "Pankaj", "Sandeep", "Tarun", "Umesh",
    "Yogesh", "Zara", "Preeti", "Swati", "Renu", "Geeta", "Sunita", "Lalita"
]

INDIAN_LAST_NAMES = [
    "Sharma", "Verma", "Gupta", "Singh", "Kumar", "Patel", "Joshi", "Mehta",
    "Reddy", "Nair", "Iyer", "Rao", "Pillai", "Menon", "Bhat", "Kaur",
    "Malhotra", "Kapoor", "Chopra", "Bose", "Das", "Sen", "Ghosh", "Roy",
    "Mishra", "Tiwari", "Pandey", "Shukla", "Dubey", "Yadav", "Agarwal",
    "Bansal", "Goel", "Jain", "Mittal", "Saxena", "Srivastava", "Tripathi"
]

INDIAN_CITIES = [
    "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Kolkata",
    "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Surat", "Nagpur",
    "Indore", "Bhopal", "Patna", "Vadodara", "Coimbatore", "Kochi",
    "Chandigarh", "Gurgaon", "Noida", "Navi Mumbai", "Thane", "Nashik"
]

CITY_TO_STATE = {
    "Mumbai": "Maharashtra", "Navi Mumbai": "Maharashtra", "Thane": "Maharashtra",
    "Pune": "Maharashtra", "Nashik": "Maharashtra", "Nagpur": "Maharashtra",
    "Delhi": "Delhi", "Gurgaon": "Haryana", "Noida": "Uttar Pradesh",
    "Bengaluru": "Karnataka", "Hyderabad": "Telangana", "Chennai": "Tamil Nadu",
    "Coimbatore": "Tamil Nadu", "Kolkata": "West Bengal", "Ahmedabad": "Gujarat",
    "Surat": "Gujarat", "Vadodara": "Gujarat", "Jaipur": "Rajasthan",
    "Lucknow": "Uttar Pradesh", "Patna": "Bihar", "Indore": "Madhya Pradesh",
    "Bhopal": "Madhya Pradesh", "Kochi": "Kerala", "Chandigarh": "Punjab"
}

# Merchant Category Codes — realistic Indian fintech context
MCC_LIST = [
    ("5411", "Grocery Stores"),
    ("5812", "Eating Places & Restaurants"),
    ("5541", "Service Stations (Fuel)"),
    ("5311", "Department Stores"),
    ("5912", "Drug Stores & Pharmacies"),
    ("7011", "Hotels & Lodging"),
    ("4111", "Local & Suburban Commuter Transport"),
    ("5732", "Electronics Stores"),
    ("5999", "Miscellaneous Retail"),
    ("5661", "Shoe Stores"),
    ("5621", "Women's Clothing Stores"),
    ("5691", "Men's & Women's Clothing"),
    ("7832", "Motion Picture Theatres"),
    ("5941", "Sporting Goods Stores"),
    ("8099", "Health & Medical Services"),
    ("8011", "Doctors & Physicians"),
    ("5045", "Computers & Peripherals"),
    ("7995", "Gambling / Betting"),   # Anomaly-prone
    ("6051", "Currency Exchange"),    # Anomaly-prone
    ("4829", "Wire Transfer"),        # Anomaly-prone
]

FOREIGN_COUNTRIES = ["USA", "UAE", "Singapore", "UK", "Germany", "China", "Thailand", "Malaysia"]

# ─────────────────────────────────────────────
# TABLE 1: internal_policies  (Agent 1)
# ─────────────────────────────────────────────

def generate_internal_policies():
    policies = [
        {
            "policy_id": "KYC-001",
            "policy_name": "Customer Onboarding & KYC Policy",
            "policy_category": "KYC",
            "regulated_entity_type": "Payment Aggregator",
            "policy_version": "v2.3",
            "effective_date": "2023-04-01",
            "last_reviewed_date": "2024-01-15",
            "policy_content": (
                "All new customers must submit Aadhaar and PAN for accounts exceeding "
                "₹10,000 monthly transaction limit. Video KYC is optional for accounts "
                "under ₹50,000 annual limit. Re-KYC required every 2 years for high-risk "
                "customers. Customer risk profiling must be completed within 7 days of "
                "onboarding. Politically Exposed Persons (PEPs) require enhanced due "
                "diligence and senior management approval."
            ),
            "status": "Active",
            "owner_team": "Compliance Team"
        },
        {
            "policy_id": "AML-001",
            "policy_name": "Anti-Money Laundering & CFT Policy",
            "policy_category": "AML",
            "regulated_entity_type": "Payment Aggregator",
            "policy_version": "v3.1",
            "effective_date": "2022-10-01",
            "last_reviewed_date": "2024-03-10",
            "policy_content": (
                "Transactions exceeding ₹10 lakhs must be reported to FIU-IND within "
                "7 working days. Suspicious Transaction Reports (STRs) must be filed "
                "within 7 days of suspicion arising. Customer Due Diligence (CDD) "
                "mandatory for all transactions above ₹50,000. Enhanced Due Diligence "
                "for cross-border transactions. Staff AML training mandatory annually. "
                "Transaction monitoring system must flag structuring patterns."
            ),
            "status": "Active",
            "owner_team": "Compliance Team"
        },
        {
            "policy_id": "AML-002",
            "policy_name": "Suspicious Transaction Monitoring Policy",
            "policy_category": "AML",
            "regulated_entity_type": "Payment Aggregator",
            "policy_version": "v1.8",
            "effective_date": "2023-01-01",
            "last_reviewed_date": "2023-11-20",
            "policy_content": (
                "Automated transaction monitoring must run on 100% of transactions. "
                "Velocity checks: more than 10 transactions in 1 hour triggers review. "
                "Round-tripping patterns (funds returning to origin within 72 hours) "
                "must be escalated. Geographic anomalies — transactions from countries "
                "on FATF grey list require manual review. ML model scores must be "
                "reviewed by human analyst for scores above 0.75."
            ),
            "status": "Active",
            "owner_team": "Risk Team"
        },
        {
            "policy_id": "LEND-001",
            "policy_name": "Digital Lending Policy",
            "policy_category": "Lending",
            "regulated_entity_type": "NBFC",
            "policy_version": "v1.2",
            "effective_date": "2023-09-01",
            "last_reviewed_date": "2024-02-01",
            "policy_content": (
                "All digital loans must be disbursed directly to borrower's bank account. "
                "Loan processing fees must be disclosed upfront before disbursement. "
                "Annual Percentage Rate (APR) must be prominently displayed. Recovery "
                "agents prohibited from contacting borrowers between 8 PM and 8 AM. "
                "Grievance redressal mechanism must be available on app and website. "
                "Credit bureau reporting mandatory within 30 days of disbursement."
            ),
            "status": "Active",
            "owner_team": "Legal"
        },
        {
            "policy_id": "LEND-002",
            "policy_name": "Credit Risk Assessment Policy",
            "policy_category": "Lending",
            "regulated_entity_type": "NBFC",
            "policy_version": "v2.0",
            "effective_date": "2023-06-01",
            "last_reviewed_date": "2024-01-30",
            "policy_content": (
                "Credit scoring models must use a minimum of 12 months transaction history. "
                "Debt-to-Income ratio must not exceed 50% for new borrowers. Bureau score "
                "below 650 requires additional collateral or co-applicant. Model performance "
                "review mandatory every quarter. Explainability report must accompany all "
                "automated credit decisions. Override by credit officer allowed with documented "
                "justification."
            ),
            "status": "Active",
            "owner_team": "Risk Team"
        },
        {
            "policy_id": "PAY-001",
            "policy_name": "Payment Aggregator Merchant Onboarding Policy",
            "policy_category": "Payments",
            "regulated_entity_type": "Payment Aggregator",
            "policy_version": "v4.0",
            "effective_date": "2022-07-01",
            "last_reviewed_date": "2024-04-01",
            "policy_content": (
                "All merchants must complete KYC before going live. Prohibited merchant "
                "categories: gambling, cryptocurrency exchanges, weapons. Escrow account "
                "mandatory — funds settlement within T+1 for verified merchants. "
                "Chargeback rate above 1% triggers enhanced monitoring. Merchant "
                "website and business model must be verified before payment link "
                "activation. Annual re-verification for all active merchants."
            ),
            "status": "Active",
            "owner_team": "Compliance Team"
        },
        {
            "policy_id": "PAY-002",
            "policy_name": "UPI Transaction Limits & Controls Policy",
            "policy_category": "Payments",
            "regulated_entity_type": "Payment Aggregator",
            "policy_version": "v2.1",
            "effective_date": "2023-03-01",
            "last_reviewed_date": "2023-12-15",
            "policy_content": (
                "UPI transaction limit: ₹1 lakh per transaction for standard users, "
                "₹2 lakhs for verified business accounts. Daily limit: ₹5 lakhs. "
                "New device cooling period: 24 hours before large transactions allowed. "
                "Two-factor authentication mandatory for transactions above ₹10,000. "
                "TPAP (Third Party Application Providers) must adhere to NPCI guidelines. "
                "Failed transaction refund must complete within 3 business days."
            ),
            "status": "Active",
            "owner_team": "Product Team"
        },
        {
            "policy_id": "GRIEV-001",
            "policy_name": "Customer Grievance Redressal Policy",
            "policy_category": "Grievance",
            "regulated_entity_type": "Payment Aggregator",
            "policy_version": "v1.5",
            "effective_date": "2023-02-01",
            "last_reviewed_date": "2024-01-10",
            "policy_content": (
                "Level 1 grievances must be acknowledged within 24 hours and resolved "
                "within 7 days. Level 2 escalations to be resolved within 15 days. "
                "Nodal Officer must be appointed and details published on website. "
                "RBI Ombudsman escalation path must be communicated to customers. "
                "Monthly grievance report to be submitted to board. "
                "Repeat complaints on same issue to be flagged as systemic risk."
            ),
            "status": "Active",
            "owner_team": "Customer Experience"
        },
        {
            "policy_id": "DATA-001",
            "policy_name": "Data Privacy & Localisation Policy",
            "policy_category": "Data Governance",
            "regulated_entity_type": "Payment Aggregator",
            "policy_version": "v1.0",
            "effective_date": "2024-01-01",
            "last_reviewed_date": "2024-06-01",
            "policy_content": (
                "All payment data must be stored in India (data localisation). "
                "Cross-border data sharing requires RBI approval. Customer consent "
                "required before sharing data with third parties. Data retention: "
                "transaction data for 5 years, KYC documents for 10 years. "
                "Right to erasure applicable for non-transaction personal data. "
                "Annual data audit by internal team with report to CISO."
            ),
            "status": "Active",
            "owner_team": "Technology"
        },
        {
            "policy_id": "FRAUD-001",
            "policy_name": "Fraud Prevention & Response Policy",
            "policy_category": "AML",
            "regulated_entity_type": "Payment Aggregator",
            "policy_version": "v3.0",
            "effective_date": "2023-05-01",
            "last_reviewed_date": "2024-05-10",
            "policy_content": (
                "Real-time fraud detection system must cover 100% of transactions. "
                "Card-not-present fraud threshold: block if ML score > 0.85. "
                "Customer notification mandatory within 30 minutes of suspected fraud. "
                "Provisional credit to customer within 10 days of fraud complaint. "
                "Fraud cases above ₹1 lakh to be reported to cybercrime cell. "
                "Post-incident review mandatory for all confirmed fraud cases above ₹5 lakhs."
            ),
            "status": "Active",
            "owner_team": "Risk Team"
        },
        {
            "policy_id": "KYC-002",
            "policy_name": "Business KYC & Merchant Verification Policy",
            "policy_category": "KYC",
            "regulated_entity_type": "Payment Aggregator",
            "policy_version": "v1.4",
            "effective_date": "2023-08-01",
            "last_reviewed_date": "2024-02-20",
            "policy_content": (
                "Business entity verification requires: GST certificate, incorporation "
                "certificate, and bank statement (last 3 months). Director KYC mandatory "
                "for all directors with >25% shareholding. Beneficial ownership declaration "
                "required for entities with complex structures. Shell company indicators: "
                "turnover below ₹10,000 with high transaction volumes triggers review. "
                "Annual refresh of business KYC for high-value merchants."
            ),
            "status": "Active",
            "owner_team": "Compliance Team"
        },
        {
            "policy_id": "LEND-003",
            "policy_name": "FLDG & Co-lending Arrangement Policy",
            "policy_category": "Lending",
            "regulated_entity_type": "NBFC",
            "policy_version": "v1.0",
            "effective_date": "2024-01-01",
            "last_reviewed_date": "2024-03-01",
            "policy_content": (
                "First Loss Default Guarantee (FLDG) capped at 5% of loan portfolio per "
                "RBI guidelines. Co-lending agreements must be disclosed to borrowers. "
                "Risk sharing between NBFC and bank partner must be documented. "
                "Minimum holding period before securitisation: 6 months. "
                "Quarterly reconciliation of co-lending portfolio mandatory. "
                "Board approval required for FLDG arrangements above ₹50 crores."
            ),
            "status": "Active",
            "owner_team": "Legal"
        },
        {
            "policy_id": "PAY-003",
            "policy_name": "Prepaid Payment Instrument (PPI) Policy",
            "policy_category": "Payments",
            "regulated_entity_type": "Payment Aggregator",
            "policy_version": "v2.2",
            "effective_date": "2022-12-01",
            "last_reviewed_date": "2023-10-01",
            "policy_content": (
                "Minimum KYC PPI: maximum balance ₹10,000, can only be used for "
                "purchase of goods and services. Full KYC PPI: maximum balance ₹2 lakhs. "
                "Cash withdrawal allowed only from full KYC PPIs. Interoperability "
                "with UPI mandatory. Expiry date must be minimum 1 year from issue. "
                "Inactive PPI: balance refund after 1 year of inactivity on request."
            ),
            "status": "Active",
            "owner_team": "Product Team"
        },
        {
            "policy_id": "RISK-001",
            "policy_name": "Liquidity Risk Management Policy",
            "policy_category": "Risk Management",
            "regulated_entity_type": "NBFC",
            "policy_version": "v1.1",
            "effective_date": "2023-11-01",
            "last_reviewed_date": "2024-04-15",
            "policy_content": (
                "Liquidity Coverage Ratio (LCR) must be maintained at minimum 100%. "
                "Net Stable Funding Ratio (NSFR) target: 110%. Stress testing quarterly "
                "covering 30-day and 90-day scenarios. Contingency Funding Plan to be "
                "reviewed annually by board. Concentration risk: no single lender to "
                "constitute more than 20% of borrowings. ALM report to RBI monthly."
            ),
            "status": "Active",
            "owner_team": "Risk Team"
        },
        {
            "policy_id": "GRIEV-002",
            "policy_name": "RBI Integrated Ombudsman Scheme Compliance Policy",
            "policy_category": "Grievance",
            "regulated_entity_type": "Payment Aggregator",
            "policy_version": "v1.0",
            "effective_date": "2023-06-01",
            "last_reviewed_date": "2024-01-05",
            "policy_content": (
                "Internal complaint resolution must be exhausted before RBI Ombudsman "
                "referral. Written rejection or no response within 30 days allows "
                "customer to escalate to Ombudsman. Nodal Officer details to be "
                "displayed on all customer-facing platforms. Annual report to RBI on "
                "grievance statistics mandatory. Compensation framework for delays in "
                "complaint resolution must be documented and published."
            ),
            "status": "Under Review",
            "owner_team": "Legal"
        }
    ]

    path = os.path.join(OUTPUT_DIR, "internal_policies.csv")
    fields = [
        "policy_id", "policy_name", "policy_category", "regulated_entity_type",
        "policy_version", "effective_date", "last_reviewed_date", "policy_content",
        "status", "owner_team"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(policies)

    print(f"✅ internal_policies.csv → {len(policies)} policies")
    return policies


# ─────────────────────────────────────────────
# TABLE 2: user_profiles  (Agent 2)
# ─────────────────────────────────────────────

def generate_user_profiles(n=60):
    profiles = []
    account_types = ["Savings", "Current", "Wallet"]
    risk_tiers = ["Low", "Low", "Low", "Medium", "Medium", "High"]  # Weighted

    mcc_codes = [m[0] for m in MCC_LIST[:12]]  # Normal MCCs only for profiles

    for i in range(n):
        uid = f"USR{str(i+1).zfill(4)}"
        first = random.choice(INDIAN_FIRST_NAMES)
        last = random.choice(INDIAN_LAST_NAMES)
        city = random.choice(INDIAN_CITIES)
        state = CITY_TO_STATE[city]

        # Spending profile — realistic INR amounts
        avg = round(random.uniform(500, 25000), 2)
        stddev = round(avg * random.uniform(0.2, 0.6), 2)

        # Each user has 3-6 favourite merchant categories
        top_mccs = random.sample(mcc_codes, k=random.randint(3, 6))

        profiles.append({
            "user_id": uid,
            "user_name": f"{first} {last}",
            "home_city": city,
            "home_state": state,
            "account_type": random.choice(account_types),
            "avg_txn_amount_90d": avg,
            "stddev_txn_amount_90d": stddev,
            "top_merchant_categories": "|".join(top_mccs),  # pipe-separated for CSV
            "active_since": rand_date(2019, 2023),
            "risk_tier": random.choice(risk_tiers)
        })

    path = os.path.join(OUTPUT_DIR, "user_profiles.csv")
    fields = [
        "user_id", "user_name", "home_city", "home_state", "account_type",
        "avg_txn_amount_90d", "stddev_txn_amount_90d", "top_merchant_categories",
        "active_since", "risk_tier"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(profiles)

    print(f"✅ user_profiles.csv → {len(profiles)} users")
    return profiles


# ─────────────────────────────────────────────
# TABLE 3: transactions  (Agent 2)
# ─────────────────────────────────────────────

ANOMALY_TYPES = ["geo_mismatch", "mcc_mismatch", "velocity_spike", "amount_outlier", "foreign_country"]

ANOMALY_REASON_TEMPLATES = {
    "geo_mismatch": (
        "Transaction originated from {txn_city} but user's registered home city is {home_city}. "
        "User has no prior transaction history in {txn_city} in the last 90 days."
    ),
    "mcc_mismatch": (
        "Merchant category {mcc_name} (MCC: {mcc}) is not in user's top merchant categories "
        "({top_mccs}). First transaction in this category in 90-day history."
    ),
    "velocity_spike": (
        "User made {count} transactions within 60 minutes, exceeding the normal hourly "
        "velocity of 2-3 transactions. Possible card testing or account takeover."
    ),
    "amount_outlier": (
        "Transaction amount ₹{amount} is {multiplier}x the user's 90-day average of "
        "₹{avg}. Statistically significant deviation (>{sigma} standard deviations)."
    ),
    "foreign_country": (
        "Transaction processed from {country}, which is outside India. User has no prior "
        "international transaction history. {country} is flagged for elevated risk monitoring."
    )
}

def build_anomaly_reason(anomaly_type, context):
    template = ANOMALY_REASON_TEMPLATES[anomaly_type]
    return template.format(**context)

def generate_transactions(profiles, n_normal=9500, n_anomaly=500):
    transactions = []
    user_map = {p["user_id"]: p for p in profiles}
    user_ids = [p["user_id"] for p in profiles]

    # ── Normal transactions ──────────────────────
    for _ in range(n_normal):
        user = user_map[random.choice(user_ids)]
        uid = user["user_id"]
        avg = float(user["avg_txn_amount_90d"])
        std = float(user["stddev_txn_amount_90d"])

        # Amount: normally distributed around user's avg
        amount = max(10, round(random.gauss(avg, std), 2))

        # MCC: pick from user's known categories (normal behaviour)
        top_mccs = user["top_merchant_categories"].split("|")
        mcc_code = random.choice(top_mccs)
        mcc_name = next((m[1] for m in MCC_LIST if m[0] == mcc_code), "Retail")

        # City: user's home city or nearby (normal)
        city = user["home_city"]
        if random.random() < 0.15:  # Occasional travel — still normal
            city = random.choice(INDIAN_CITIES)

        transactions.append({
            "transaction_id": f"TXN{str(len(transactions)+1).zfill(6)}",
            "user_id": uid,
            "timestamp": rand_datetime(),
            "amount": amount,
            "currency": "INR",
            "merchant_name": f"{mcc_name} Store {random.randint(1, 999)}",
            "merchant_category_code": mcc_code,
            "merchant_city": city,
            "merchant_country": "India",
            "channel": random.choice(["UPI", "UPI", "UPI", "Card", "Netbanking", "Wallet"]),
            "device_id": f"DEV{uid}{random.randint(1,3)}",
            "ip_country": "India",
            "is_anomaly": False,
            "anomaly_type": None,
            "anomaly_reason": None
        })

    # ── Anomalous transactions ───────────────────
    velocity_users = {}  # Track velocity spike users

    for i in range(n_anomaly):
        user = user_map[random.choice(user_ids)]
        uid = user["user_id"]
        avg = float(user["avg_txn_amount_90d"])
        std = float(user["stddev_txn_amount_90d"])
        home_city = user["home_city"]
        top_mccs = user["top_merchant_categories"].split("|")

        anomaly_type = random.choice(ANOMALY_TYPES)

        # Build transaction based on anomaly type
        if anomaly_type == "geo_mismatch":
            # Transaction from a city the user has never been to
            other_cities = [c for c in INDIAN_CITIES if c != home_city]
            txn_city = random.choice(other_cities)
            amount = round(random.gauss(avg, std * 0.5), 2)
            amount = max(100, amount)
            mcc_code = random.choice(top_mccs)
            mcc_name = next((m[1] for m in MCC_LIST if m[0] == mcc_code), "Retail")
            reason = build_anomaly_reason("geo_mismatch", {
                "txn_city": txn_city, "home_city": home_city
            })
            row = {
                "merchant_city": txn_city, "merchant_country": "India",
                "merchant_category_code": mcc_code,
                "merchant_name": f"{mcc_name} Store {random.randint(1,999)}",
                "amount": amount, "ip_country": "India"
            }

        elif anomaly_type == "mcc_mismatch":
            # Transaction in a category user never uses
            all_mccs = [m[0] for m in MCC_LIST]
            unusual_mccs = [m for m in all_mccs if m not in top_mccs]
            mcc_code = random.choice(unusual_mccs) if unusual_mccs else "7995"
            mcc_name = next((m[1] for m in MCC_LIST if m[0] == mcc_code), "Unknown")
            amount = round(random.gauss(avg, std * 0.5), 2)
            amount = max(100, amount)
            reason = build_anomaly_reason("mcc_mismatch", {
                "mcc": mcc_code, "mcc_name": mcc_name,
                "top_mccs": ", ".join(top_mccs[:3])
            })
            row = {
                "merchant_city": home_city, "merchant_country": "India",
                "merchant_category_code": mcc_code,
                "merchant_name": f"{mcc_name} Store {random.randint(1,999)}",
                "amount": amount, "ip_country": "India"
            }

        elif anomaly_type == "velocity_spike":
            # High transaction count — we just mark this one as part of a spike
            count = random.randint(8, 20)
            amount = round(avg * random.uniform(0.3, 1.5), 2)
            mcc_code = random.choice(top_mccs)
            mcc_name = next((m[1] for m in MCC_LIST if m[0] == mcc_code), "Retail")
            reason = build_anomaly_reason("velocity_spike", {"count": count})
            row = {
                "merchant_city": home_city, "merchant_country": "India",
                "merchant_category_code": mcc_code,
                "merchant_name": f"{mcc_name} Store {random.randint(1,999)}",
                "amount": amount, "ip_country": "India"
            }

        elif anomaly_type == "amount_outlier":
            # Amount is 4-10x the user's average
            multiplier = round(random.uniform(4, 10), 1)
            amount = round(avg * multiplier, 2)
            sigma = round((amount - avg) / max(std, 1), 1)
            mcc_code = random.choice(top_mccs)
            mcc_name = next((m[1] for m in MCC_LIST if m[0] == mcc_code), "Retail")
            reason = build_anomaly_reason("amount_outlier", {
                "amount": f"{amount:,.2f}",
                "multiplier": multiplier,
                "avg": f"{avg:,.2f}",
                "sigma": sigma
            })
            row = {
                "merchant_city": home_city, "merchant_country": "India",
                "merchant_category_code": mcc_code,
                "merchant_name": f"{mcc_name} Store {random.randint(1,999)}",
                "amount": amount, "ip_country": "India"
            }

        else:  # foreign_country
            country = random.choice(FOREIGN_COUNTRIES)
            amount = round(avg * random.uniform(1.5, 5), 2)
            mcc_code = random.choice(["7011", "5812", "5311"])  # Hotel/Restaurant/Store
            mcc_name = next((m[1] for m in MCC_LIST if m[0] == mcc_code), "Retail")
            reason = build_anomaly_reason("foreign_country", {"country": country})
            row = {
                "merchant_city": random.choice(["New York", "Dubai", "Singapore", "London"]),
                "merchant_country": country,
                "merchant_category_code": mcc_code,
                "merchant_name": f"{mcc_name} {random.randint(1,99)}",
                "amount": amount, "ip_country": country
            }

        transactions.append({
            "transaction_id": f"TXN{str(len(transactions)+1).zfill(6)}",
            "user_id": uid,
            "timestamp": rand_datetime(),
            "amount": row["amount"],
            "currency": "INR",
            "merchant_name": row["merchant_name"],
            "merchant_category_code": row["merchant_category_code"],
            "merchant_city": row["merchant_city"],
            "merchant_country": row["merchant_country"],
            "channel": random.choice(["UPI", "Card", "Netbanking", "Wallet"]),
            "device_id": f"DEV{uid}{random.randint(1,5)}",
            "ip_country": row["ip_country"],
            "is_anomaly": True,
            "anomaly_type": anomaly_type,
            "anomaly_reason": reason
        })

    # Shuffle so anomalies aren't all at the end
    random.shuffle(transactions)

    # Re-assign sequential transaction IDs after shuffle
    for idx, txn in enumerate(transactions):
        txn["transaction_id"] = f"TXN{str(idx+1).zfill(6)}"

    path = os.path.join(OUTPUT_DIR, "transactions.csv")
    fields = [
        "transaction_id", "user_id", "timestamp", "amount", "currency",
        "merchant_name", "merchant_category_code", "merchant_city",
        "merchant_country", "channel", "device_id", "ip_country",
        "is_anomaly", "anomaly_type", "anomaly_reason"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(transactions)

    anomaly_count = sum(1 for t in transactions if t["is_anomaly"])
    print(f"✅ transactions.csv → {len(transactions)} total | {anomaly_count} anomalies ({round(anomaly_count/len(transactions)*100,1)}%)")

    # Print anomaly breakdown
    from collections import Counter
    types = Counter(t["anomaly_type"] for t in transactions if t["is_anomaly"])
    for k, v in types.items():
        print(f"   ↳ {k}: {v}")

    return transactions


# ─────────────────────────────────────────────
# TABLE 4: clients  (Agent 3)
# ─────────────────────────────────────────────

def generate_clients(n=30):
    sectors = ["NBFC", "NBFC", "Insurance", "Retail Fintech", "Retail Fintech",
               "Microfinance", "Housing Finance", "Payment Gateway", "Wealth Management"]

    product_types = {
        "NBFC": ["Retail Lending", "Working Capital", "MSME Loans"],
        "Insurance": ["Premium Collection", "Claims Settlement"],
        "Retail Fintech": ["Payment Settlement", "Buy Now Pay Later", "Digital Wallet"],
        "Microfinance": ["Micro Lending", "Group Loans"],
        "Housing Finance": ["Home Loans", "Loan Against Property"],
        "Payment Gateway": ["Payment Settlement", "Escrow Services"],
        "Wealth Management": ["Portfolio Custody", "Mutual Fund Distribution"]
    }

    company_prefixes = [
        "Apex", "Bharat", "Capital", "Dhruv", "Eterna", "Finway", "Growfast",
        "Horizon", "India", "Jaya", "Kiran", "Lakshmi", "Mahesh", "Nidhi",
        "Omni", "Param", "Quantum", "Ratna", "Sahara", "Taara", "Udaan",
        "Vaibhav", "Wealthwise", "Xceed", "Yatra", "Zenith"
    ]
    company_suffixes = ["Finance", "Capital", "Fintech", "Credit", "Money", "Ventures", "Solutions"]

    clients = []
    used_names = set()

    for i in range(n):
        cid = f"CLT{str(i+1).zfill(3)}"
        sector = random.choice(sectors)

        # Unique company name
        while True:
            name = f"{random.choice(company_prefixes)} {random.choice(company_suffixes)}"
            if name not in used_names:
                used_names.add(name)
                break

        product = random.choice(product_types.get(sector, ["Financial Services"]))

        # Exposure amounts — realistic INR crores (stored in lakhs for precision)
        credit_total = round(random.uniform(500, 15000) * 100000, 0)  # ₹50L to ₹150Cr
        utilized_pct = round(random.uniform(30, 95), 1)
        short_term_pct = round(random.uniform(30, 80), 1)
        short_term = round(credit_total * short_term_pct / 100, 0)
        long_term = round(credit_total - short_term, 0)

        risk_rating = random.choices(
            ["Low", "Medium", "High"],
            weights=[40, 40, 20]
        )[0]

        reg_sensitivity = {
            "NBFC": "High", "Microfinance": "High", "Housing Finance": "High",
            "Insurance": "High", "Payment Gateway": "Medium",
            "Retail Fintech": "Medium", "Wealth Management": "Low"
        }.get(sector, "Medium")

        payment_history = random.choices(
            ["Clean", "Minor Delays", "Major Delays"],
            weights=[60, 30, 10]
        )[0]

        clients.append({
            "client_id": cid,
            "client_name": name,
            "sector": sector,
            "product_type": product,
            "exposure_short_term": short_term,
            "exposure_long_term": long_term,
            "credit_line_total": credit_total,
            "credit_line_utilized_pct": utilized_pct,
            "collateral_type": random.choice(["Secured", "Secured", "Unsecured"]),
            "risk_rating": risk_rating,
            "last_reviewed_date": rand_date_recent(),
            "payment_history": payment_history,
            "regulatory_sensitivity": reg_sensitivity
        })

    path = os.path.join(OUTPUT_DIR, "clients.csv")
    fields = [
        "client_id", "client_name", "sector", "product_type",
        "exposure_short_term", "exposure_long_term", "credit_line_total",
        "credit_line_utilized_pct", "collateral_type", "risk_rating",
        "last_reviewed_date", "payment_history", "regulatory_sensitivity"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(clients)

    print(f"✅ clients.csv → {len(clients)} clients")

    # Print sector breakdown
    from collections import Counter
    sector_counts = Counter(c["sector"] for c in clients)
    for s, count in sector_counts.most_common():
        total_exp = sum(
            c["credit_line_total"] for c in clients if c["sector"] == s
        )
        print(f"   ↳ {s}: {count} clients | ₹{total_exp/10000000:.1f}Cr exposure")

    return clients


# ─────────────────────────────────────────────
# TABLE 5: sector_news_mapping  (Agent 3)
# ─────────────────────────────────────────────

def generate_sector_news_mapping():
    mapping = [
        {
            "sector": "NBFC",
            "keywords": [
                "NBFC", "non-banking financial", "shadow banking",
                "liquidity norms", "NBFC regulation", "non-bank lender",
                "NBFC-MFI", "systemically important NBFC", "layer structure"
            ],
            "related_regulators": ["RBI", "SEBI"]
        },
        {
            "sector": "Insurance",
            "keywords": [
                "insurance", "IRDAI", "insurer", "premium", "life insurance",
                "general insurance", "health insurance", "reinsurance",
                "insurance regulation", "surrender value", "policy surrender"
            ],
            "related_regulators": ["IRDAI", "SEBI"]
        },
        {
            "sector": "Retail Fintech",
            "keywords": [
                "fintech", "digital payments", "payment aggregator", "UPI",
                "digital lending", "BNPL", "buy now pay later", "neobank",
                "digital wallet", "prepaid payment", "payment gateway"
            ],
            "related_regulators": ["RBI", "NPCI"]
        },
        {
            "sector": "Microfinance",
            "keywords": [
                "microfinance", "MFI", "NBFC-MFI", "self-help group",
                "group lending", "microcredit", "priority sector lending",
                "financial inclusion", "rural credit", "SHG"
            ],
            "related_regulators": ["RBI", "NABARD"]
        },
        {
            "sector": "Housing Finance",
            "keywords": [
                "housing finance", "HFC", "home loan", "mortgage",
                "NHB", "National Housing Bank", "affordable housing",
                "real estate lending", "loan against property", "LAP"
            ],
            "related_regulators": ["NHB", "RBI"]
        },
        {
            "sector": "Payment Gateway",
            "keywords": [
                "payment gateway", "payment aggregator", "PPI", "prepaid",
                "escrow", "nodal account", "merchant payments",
                "card network", "transaction switching"
            ],
            "related_regulators": ["RBI", "NPCI"]
        },
        {
            "sector": "Wealth Management",
            "keywords": [
                "wealth management", "mutual fund", "portfolio management",
                "PMS", "AIF", "alternative investment", "SEBI regulations",
                "investment adviser", "robo advisory", "asset management"
            ],
            "related_regulators": ["SEBI", "AMFI"]
        }
    ]

    path = os.path.join(OUTPUT_DIR, "sector_news_mapping.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"✅ sector_news_mapping.json → {len(mapping)} sectors mapped")
    return mapping


# ─────────────────────────────────────────────
# TABLE 6: audit_logs  (Agent 4 — schema only)
# ─────────────────────────────────────────────

def generate_audit_logs_schema():
    """
    Audit logs are NOT pre-populated — they're generated at runtime
    by the other three agents. We create an empty CSV with the correct
    schema so PostgreSQL migration and the dashboard can reference it.
    """
    path = os.path.join(OUTPUT_DIR, "audit_logs.csv")
    fields = [
        "log_id", "agent_name", "trigger_type", "input_summary",
        "output_summary", "full_reasoning", "confidence_score",
        "action_taken", "timestamp", "related_entity_id",
        "related_entity_type", "query_text"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        # No rows — populated at runtime

    print(f"✅ audit_logs.csv → empty schema (12 columns, populated at runtime by agents)")


# ─────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────

def print_summary(policies, profiles, transactions, clients):
    print("\n" + "="*60)
    print("  FINORA SYNTHETIC DATA — GENERATION COMPLETE")
    print("="*60)
    print(f"\n📁 Output directory: {os.path.abspath(OUTPUT_DIR)}/")
    print()
    print("  FILE                      ROWS    AGENT")
    print("  ─────────────────────────────────────────────")
    print(f"  internal_policies.csv     {len(policies):<7} Agent 1 — Regulatory Watch")
    print(f"  user_profiles.csv         {len(profiles):<7} Agent 2 — Transaction Risk")
    print(f"  transactions.csv          {len(transactions):<7} Agent 2 — Transaction Risk")
    print(f"  clients.csv               {len(clients):<7} Agent 3 — Market Sentiment")
    print(f"  sector_news_mapping.json  7       Agent 3 — Market Sentiment")
    print(f"  audit_logs.csv            0       Agent 4 — Audit Trail (runtime)")
    print()
    print("  WHAT GOES WHERE:")
    print("  ─────────────────────────────────────────────")
    print("  PostgreSQL  → all CSVs (structured queries)")
    print("  Qdrant      → internal_policies.policy_content (embedded)")
    print("              → regulation PDFs you download separately")
    print()
    anomalies = [t for t in transactions if t["is_anomaly"]]
    print(f"  ML MODEL TRAINING NOTE:")
    print(f"  {len(transactions)} transactions | {len(anomalies)} anomalies ({round(len(anomalies)/len(transactions)*100,1)}%)")
    print(f"  Use is_anomaly as label for Isolation Forest + XGBoost")
    print(f"  anomaly_reason = ground truth for LLM explanation eval")
    print()
    print("  NEXT STEPS:")
    print("  1. Load CSVs into PostgreSQL (use SQLAlchemy or psycopg2)")
    print("  2. Chunk + embed internal_policies into Qdrant")
    print("  3. Download 10-15 real RBI/SEBI PDFs → /data/regulations/")
    print("  4. Train ML model on transactions.csv")
    print("  5. Build Agent 2 first — richest data, best demo impact")
    print("="*60 + "\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🚀 FINORA Synthetic Data Generator")
    print("─" * 40)

    policies   = generate_internal_policies()
    profiles   = generate_user_profiles(n=60)
    txns       = generate_transactions(profiles, n_normal=9500, n_anomaly=500)
    clients    = generate_clients(n=30)
    _          = generate_sector_news_mapping()
    _          = generate_audit_logs_schema()

    print_summary(policies, profiles, txns, clients)