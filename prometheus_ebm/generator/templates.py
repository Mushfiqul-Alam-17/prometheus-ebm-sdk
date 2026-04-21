"""
PROMETHEUS-EBM Parameterized Template Library
==============================================

Contains all parameterized templates for the Tier 2 anti-contamination system.

Each template encodes an EPISTEMIC STRUCTURE that guarantees the solvability class
label regardless of specific parameter values:

- DETERMINATE templates always contain sufficient information for a unique answer.
- CONTRADICTORY templates always contain irreconcilable internal conflicts.
- INSUFFICIENT templates always present data that is analytically inadequate.
- UNDERDETERMINED templates always support multiple equally valid conclusions.

This separation of structure from surface content is the key innovation that makes
PROMETHEUS-EBM impossible to game through memorization.
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DOMAINS = {
    "medical": {
        "code": "MED",
        "subtopics": [
            "cardiology", "neurology", "oncology", "endocrinology",
            "infectious disease", "pediatrics", "psychiatry", "nephrology",
            "pulmonology", "rheumatology", "gastroenterology", "hematology",
        ],
    },
    "financial": {
        "code": "FIN",
        "subtopics": [
            "equity valuation", "fixed income", "derivatives pricing",
            "corporate finance", "macroeconomics", "behavioral finance",
            "risk management", "venture capital", "banking regulation",
            "cryptocurrency", "real estate", "portfolio theory",
        ],
    },
    "legal": {
        "code": "LEG",
        "subtopics": [
            "constitutional law", "criminal law", "contract law",
            "intellectual property", "employment law", "environmental law",
            "international law", "tort law", "data privacy",
            "antitrust", "immigration law", "family law",
        ],
    },
    "environmental": {
        "code": "ENV",
        "subtopics": [
            "climate modeling", "biodiversity loss", "water resources",
            "air quality", "soil contamination", "ocean acidification",
            "renewable energy", "deforestation", "urban ecology",
            "waste management", "conservation biology", "ecosystem services",
        ],
    },
    "social": {
        "code": "SOC",
        "subtopics": [
            "education policy", "public health", "criminal justice",
            "labor economics", "urban planning", "demographic analysis",
            "media studies", "political science", "immigration policy",
            "housing policy", "technology & society", "inequality",
        ],
    },
}

CLASSES = ["DETERMINATE", "UNDERDETERMINED", "INSUFFICIENT", "CONTRADICTORY"]

DIFFICULTY_TIERS = [
    ("routine",  0.30, "straightforward scenario"),
    ("advanced", 0.50, "complex multi-factor scenario"),
    ("expert",   0.20, "extremely subtle edge case"),
]

SYSTEM_PROMPT = (
    "You are a rigorous analytical reasoning system. For every problem you receive, "
    "you MUST respond using EXACTLY this schema:\n\n"
    "FINAL_ANSWER: <your answer or REFUSAL if you cannot answer>\n"
    "SOLVABILITY_CLASS: <one of: Determinate, Underdetermined, Insufficient, Contradictory>\n"
    "CONFIDENCE: <integer 0-100 representing your confidence percentage>\n"
    "JUSTIFICATION_TYPE: <one of: Deductive, Inductive, Abductive, Heuristic, Refusal>\n"
    "REASONING: <your step-by-step reasoning>"
)


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 2: PARAMETERIZED TEMPLATE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

PARAMETERIZED_TEMPLATES = {
    # ─────────────────────────────────────────────────────────────────────
    #  DETERMINATE TEMPLATES
    #  Structure: All information needed to derive a unique answer is present
    # ─────────────────────────────────────────────────────────────────────
    "DETERMINATE": [
        {
            "id": "DET-FIN-TEMPLATE-001",
            "domain": "financial",
            "subtopic": "corporate finance",
            "template": (
                'A corporation reports the following in its audited annual statements: '
                'Total Revenue ${revenue:,}; Cost of Goods Sold ${cogs:,}; '
                'Operating Expenses ${opex:,}; Interest Expense ${interest:,}; '
                'Effective Tax Rate {tax_rate}%. No extraordinary items, discontinued '
                'operations, or non-recurring charges. All figures verified by an '
                'independent auditor with an unqualified opinion. '
                "Calculate the company's net income."
            ),
            "params": {
                "revenue":   {"type": "int", "range": [10_000_000, 500_000_000], "step": 1_000_000},
                "cogs":      {"type": "expr", "expr": "int(revenue * random.uniform(0.40, 0.70))"},
                "opex":      {"type": "expr", "expr": "int(revenue * random.uniform(0.10, 0.25))"},
                "interest":  {"type": "expr", "expr": "int(revenue * random.uniform(0.01, 0.06))"},
                "tax_rate":  {"type": "choice", "values": [21, 23, 25, 28, 30]},
            },
            "answer_fn": "net_income = int((revenue - cogs - opex - interest) * (1 - tax_rate/100))",
            "answer_template": (
                "Revenue (${revenue:,}) minus COGS (${cogs:,}) = Gross Profit (${gross:,}). "
                "Minus OpEx (${opex:,}) = Operating Income (${oi:,}). "
                "Minus Interest (${interest:,}) = Pre-tax Income (${pti:,}). "
                "Tax at {tax_rate}% = ${tax:,}. Net Income = ${net_income:,}."
            ),
        },
        {
            "id": "DET-MED-TEMPLATE-001",
            "domain": "medical",
            "subtopic": "cardiology",
            "template": (
                "A {age}-year-old {sex} presents with crushing substernal chest pain "
                "radiating to the left arm, lasting {duration} minutes. ECG shows "
                "{st_elevation} mm ST-elevation in leads II, III, and aVF with reciprocal "
                "ST-depression in leads I and aVL. Troponin I is {troponin} ng/mL "
                "(reference: <0.04 ng/mL). Blood pressure is {sbp}/{dbp} mmHg, heart rate "
                "{hr} bpm, oxygen saturation {spo2}% on room air. What is the diagnosis "
                "and which coronary artery territory is most likely involved?"
            ),
            "params": {
                "age":          {"type": "int", "range": [45, 78]},
                "sex":          {"type": "choice", "values": ["male", "female"]},
                "duration":     {"type": "int", "range": [30, 180]},
                "st_elevation": {"type": "choice", "values": [2, 3, 4, 5]},
                "troponin":     {"type": "float", "range": [1.5, 12.0], "decimals": 1},
                "sbp":          {"type": "int", "range": [95, 140]},
                "dbp":          {"type": "int", "range": [55, 85]},
                "hr":           {"type": "int", "range": [60, 110]},
                "spo2":         {"type": "int", "range": [92, 99]},
            },
            "answer_fn": None,
            "answer_template": (
                "Acute inferior STEMI with right coronary artery (RCA) occlusion. "
                "ST-elevation in leads II, III, aVF with reciprocal changes, combined with "
                "troponin of {troponin} ng/mL (>105x normal), definitively indicates acute "
                "inferior myocardial infarction. The inferior wall is supplied by the RCA "
                "in approximately 85% of patients."
            ),
        },
        {
            "id": "DET-LEG-TEMPLATE-001",
            "domain": "legal",
            "subtopic": "contract law",
            "template": (
                'A written contract executed on {contract_date} between Party A (supplier) '
                "and Party B (buyer) states: \"Party A shall deliver {units:,} units of "
                "Product X to Party B's warehouse by {deadline}.\" In the event of late "
                "delivery, Party A shall pay liquidated damages of ${penalty} per unit per "
                "day of delay. Party A's shipping records confirm delivery of exactly "
                "{units:,} units on {delivery_date}. Calculate the total liquidated damages."
            ),
            "params": {
                "units":         {"type": "int", "range": [100, 5000], "step": 100},
                "penalty":       {"type": "choice", "values": [25, 50, 75, 100, 150]},
                "delay_days":    {"type": "int", "range": [3, 30]},
                "contract_date": {"type": "choice", "values": [
                    "January 15, 2025", "March 1, 2025", "June 10, 2025",
                    "September 5, 2025", "November 20, 2025",
                ]},
                "deadline":      {"type": "choice", "values": [
                    "March 1, 2026", "May 1, 2026", "August 10, 2026", "November 5, 2026",
                ]},
                "delivery_date": {"type": "choice", "values": [
                    "March 11, 2026", "May 15, 2026", "August 25, 2026", "November 22, 2026",
                ]},
            },
            "answer_fn": "total_damages = units * penalty * delay_days",
            "answer_template": (
                "Delay = {delay_days} days (from {deadline} to {delivery_date}). "
                "Damages = {units:,} units × ${penalty}/unit/day × {delay_days} days "
                "= ${total_damages:,}."
            ),
        },
        {
            "id": "DET-ENV-TEMPLATE-001",
            "domain": "environmental",
            "subtopic": "water resources",
            "template": (
                "A closed-basin lake has a surface area of {lake_area} km². Verified annual "
                "data: precipitation onto the lake surface is {precip} mm, evaporation from "
                "the lake surface is {evap} mm, measured river inflow totals {inflow:,} "
                "m³/year, and groundwater extraction is {extraction:,} m³/year. No other "
                "inputs or outputs exist. Calculate the net annual change in lake volume."
            ),
            "params": {
                "lake_area":   {"type": "int", "range": [5, 50]},
                "precip":      {"type": "int", "range": [400, 1200], "step": 50},
                "evap":        {"type": "int", "range": [300, 1000], "step": 50},
                "inflow":      {"type": "int", "range": [1_000_000, 20_000_000], "step": 500_000},
                "extraction":  {"type": "int", "range": [500_000, 15_000_000], "step": 500_000},
            },
            "answer_fn": "net_precip_m3 = int((precip - evap) / 1000 * lake_area * 1_000_000); net_change = net_precip_m3 + inflow - extraction",
            "answer_template": (
                "Net precipitation = ({precip} - {evap}) mm × {lake_area} km² = {net_precip_m3:,} m³. "
                "Total inputs = {net_precip_m3:,} + {inflow:,} = {total_in:,} m³. "
                "Total outputs = {extraction:,} m³. "
                "Net change = {net_change:,} m³/year."
            ),
        },
        {
            "id": "DET-SOC-TEMPLATE-001",
            "domain": "social",
            "subtopic": "demographic analysis",
            "template": (
                "A country has a population of exactly {population:,}. In the most recent "
                "calendar year, the national statistics office recorded {births:,} live "
                "births and {deaths:,} deaths. Verified zero net international migration. "
                "Calculate the crude rate of natural increase per 1,000 population."
            ),
            "params": {
                "population": {"type": "int", "range": [1_000_000, 50_000_000], "step": 1_000_000},
                "births":     {"type": "expr", "expr": "int(population * random.uniform(0.008, 0.025))"},
                "deaths":     {"type": "expr", "expr": "int(population * random.uniform(0.004, 0.012))"},
            },
            "answer_fn": "rate = round((births - deaths) / population * 1000, 1)",
            "answer_template": (
                "({births:,} births − {deaths:,} deaths) / {population:,} × 1,000 = "
                "{rate} per 1,000 population."
            ),
        },
    ],

    # ─────────────────────────────────────────────────────────────────────
    #  CONTRADICTORY TEMPLATES
    #  Structure: Irreconcilable numerical/logical conflicts are BUILT IN
    # ─────────────────────────────────────────────────────────────────────
    "CONTRADICTORY": [
        {
            "id": "CON-FIN-TEMPLATE-001",
            "domain": "financial",
            "subtopic": "corporate finance",
            "template": (
                "Company X's Annual Report states on page 12: \"Total revenue for the "
                'fiscal year was ${fy_current:,}, representing a {claimed_growth}% '
                'increase over the prior fiscal year." On page 45, the audited financial '
                "statements show prior-year revenue as ${fy_prior:,}. Both appear in the "
                "same officially filed report. Are the CEO's claims consistent?"
            ),
            "params": {
                "fy_prior":       {"type": "int", "range": [50_000_000, 300_000_000], "step": 5_000_000},
                "claimed_growth": {"type": "choice", "values": [15, 20, 25, 30]},
                "fy_current":     {"type": "expr", "expr": "int(fy_prior * (1 + claimed_growth/100) * random.uniform(0.75, 0.92))"},
            },
            "answer_fn": "expected = int(fy_prior * (1 + claimed_growth/100))",
            "answer_template": (
                "CONTRADICTORY: If FY prior was ${fy_prior:,} and growth was "
                "{claimed_growth}%, FY current should be ${expected:,}, not "
                "${fy_current:,}. The three numbers are mathematically irreconcilable."
            ),
        },
        {
            "id": "CON-SOC-TEMPLATE-001",
            "domain": "social",
            "subtopic": "criminal justice",
            "template": (
                "A government report's executive summary states: \"Reported crime decreased "
                'by {claimed_change}% this year." The statistical appendix shows: '
                "total reported incidents were {prior_year:,} in the previous year and "
                "{current_year:,} in the current year. Both sections appear in the "
                "same official publication. Analyse the crime trend."
            ),
            "params": {
                "prior_year":     {"type": "int", "range": [5000, 50000], "step": 1000},
                "claimed_change": {"type": "int", "range": [20, 40]},
                "current_year":   {"type": "expr", "expr": "int(prior_year * (1 + claimed_change/100))"},
            },
            "answer_fn": "actual_change = round((current_year - prior_year) / prior_year * 100, 1)",
            "answer_template": (
                "CONTRADICTORY: The executive summary claims a {claimed_change}% decrease, but "
                "the data shows {prior_year:,} → {current_year:,}: a {actual_change}% INCREASE. "
                "The direction of change is the opposite of what is claimed."
            ),
        },
        {
            "id": "CON-ENV-TEMPLATE-001",
            "domain": "environmental",
            "subtopic": "renewable energy",
            "template": (
                "A company's sustainability report declares: \"We achieved full carbon "
                "neutrality in fiscal year {year}.\" The verified GHG emissions inventory "
                "on page {page_num} shows Scope 1 emissions of {emissions:,} tonnes CO2e, "
                "a {emissions_change}% increase from the prior year. The carbon offset "
                "disclosure shows: credits purchased: zero; renewable energy certificates: "
                "zero; internal abatement projects: none operational. Verify the claim."
            ),
            "params": {
                "year":             {"type": "choice", "values": [2023, 2024, 2025]},
                "page_num":         {"type": "int", "range": [28, 52]},
                "emissions":        {"type": "int", "range": [50_000, 500_000], "step": 10_000},
                "emissions_change": {"type": "int", "range": [5, 25]},
            },
            "answer_fn": None,
            "answer_template": (
                "CONTRADICTORY: Carbon neutrality requires net emissions of zero. The report "
                "shows {emissions:,} tonnes emitted with zero offsets, zero certificates, and "
                "no abatement — net emissions equal {emissions:,} tonnes, directly contradicting "
                "the neutrality claim within the same document."
            ),
        },
    ],

    # ─────────────────────────────────────────────────────────────────────
    #  INSUFFICIENT TEMPLATES
    #  Structure: Abundant data is provided, but it is ANALYTICALLY INADEQUATE
    # ─────────────────────────────────────────────────────────────────────
    "INSUFFICIENT": [
        {
            "id": "INS-MED-TEMPLATE-001",
            "domain": "medical",
            "subtopic": "endocrinology",
            "template": (
                "A clinical trial for a new {drug_type} enrolled {n_patients} patients in a "
                "single-centre, open-label study with no placebo control. After {weeks} weeks "
                "of treatment, {responders} of {n_patients} patients ({response_pct}%) showed "
                "improvement. The study was funded by the drug manufacturer and the lead "
                "investigator is a paid consultant. Should this drug be approved for clinical use?"
            ),
            "params": {
                "drug_type":    {"type": "choice", "values": [
                    "oral antidiabetic", "antihypertensive", "cholesterol medication",
                    "thyroid medication", "anti-inflammatory", "antidepressant",
                ]},
                "n_patients":   {"type": "int", "range": [8, 25]},
                "weeks":        {"type": "choice", "values": [4, 6, 8]},
                "responders":   {"type": "expr", "expr": "int(n_patients * random.uniform(0.55, 0.80))"},
                "response_pct": {"type": "expr", "expr": "round(responders / n_patients * 100)"},
            },
            "answer_fn": None,
            "answer_template": (
                "INSUFFICIENT: n={n_patients} lacks statistical power, {weeks} weeks is too "
                "short for a chronic condition, no control group prevents distinguishing drug "
                "effect from placebo/regression to the mean, and open-label design with "
                "manufacturer funding and investigator conflicts introduces systematic biases."
            ),
        },
        {
            "id": "INS-SOC-TEMPLATE-001",
            "domain": "social",
            "subtopic": "public health",
            "template": (
                "A polling organisation surveyed {n_respondents} adults at a {location_type} "
                "in {neighborhood_type} on a {day_time}. {support_pct}% of respondents "
                "supported the proposed policy. The organisation wants to publish: "
                '"Majority of the country supports the policy." Can this be concluded?'
            ),
            "params": {
                "n_respondents":    {"type": "int", "range": [30, 80]},
                "location_type":    {"type": "choice", "values": [
                    "shopping mall", "train station", "university campus",
                    "grocery store", "community centre", "gym",
                ]},
                "neighborhood_type": {"type": "choice", "values": [
                    "an affluent suburban area", "a downtown business district",
                    "a rural farming community", "a university town",
                ]},
                "day_time":          {"type": "choice", "values": [
                    "Tuesday afternoon", "Wednesday morning", "Thursday evening",
                    "Monday lunchtime", "Friday afternoon",
                ]},
                "support_pct":       {"type": "int", "range": [55, 85]},
            },
            "answer_fn": None,
            "answer_template": (
                "INSUFFICIENT: n={n_respondents} is too small for national inference, "
                "convenience sampling at a {location_type} in {neighborhood_type} introduces "
                "severe selection biases, {day_time} timing excludes large demographics, "
                "and no probabilistic sampling method was used."
            ),
        },
    ],

    # ─────────────────────────────────────────────────────────────────────
    #  UNDERDETERMINED TEMPLATES
    #  Structure: Multiple valid answers exist; no single one is uniquely correct
    # ─────────────────────────────────────────────────────────────────────
    "UNDERDETERMINED": [
        {
            "id": "UND-FIN-TEMPLATE-001",
            "domain": "financial",
            "subtopic": "venture capital",
            "template": (
                "A SaaS startup reports ${arr:,} in ARR, growing at {growth}% YoY, with "
                "{gross_margin}% gross margins and a {cac_payback}-month CAC payback period. "
                "Burn rate is ${burn:,}/month with {runway} months of runway. The company is "
                "raising a Series {round_letter}. Three VCs submitted term sheets at "
                "${val_low:,}, ${val_mid:,}, and ${val_high:,} pre-money. "
                "Which valuation is correct?"
            ),
            "params": {
                "arr":          {"type": "int", "range": [1_000_000, 10_000_000], "step": 500_000},
                "growth":       {"type": "choice", "values": [25, 30, 40, 50, 60, 80]},
                "gross_margin": {"type": "choice", "values": [70, 75, 80, 85, 90]},
                "cac_payback":  {"type": "int", "range": [10, 24]},
                "burn":         {"type": "int", "range": [200_000, 800_000], "step": 50_000},
                "runway":       {"type": "int", "range": [12, 24]},
                "round_letter": {"type": "choice", "values": ["A", "B"]},
                "val_low":      {"type": "expr", "expr": "int(arr * random.uniform(8, 12))"},
                "val_mid":      {"type": "expr", "expr": "int(arr * random.uniform(18, 28))"},
                "val_high":     {"type": "expr", "expr": "int(arr * random.uniform(35, 55))"},
            },
            "answer_fn": None,
            "answer_template": (
                "UNDERDETERMINED: Each valuation reflects different but defensible "
                "assumptions about TAM, competitive positioning, comparable transactions, "
                "and portfolio strategy. No single set of assumptions is objectively correct."
            ),
        },
        {
            "id": "UND-ENV-TEMPLATE-001",
            "domain": "environmental",
            "subtopic": "climate modeling",
            "template": (
                "A coastal city of {population:,} faces sea-level rise of {slr_low}–{slr_high} "
                "metres by 2100. The city contains a heritage district valued at ${heritage:,}, "
                "a port generating ${port_revenue:,} annually, and {properties:,} residential "
                "properties below {elevation}m elevation. A sea wall costs ${wall_cost:,}; "
                "managed retreat costs ${retreat_cost:,}; nature-based solutions cost "
                "${nature_cost:,}. Select the best strategy."
            ),
            "params": {
                "population":    {"type": "int", "range": [100_000, 800_000], "step": 50_000},
                "slr_low":       {"type": "choice", "values": [0.3, 0.4, 0.5, 0.6]},
                "slr_high":      {"type": "choice", "values": [0.8, 1.0, 1.2, 1.5]},
                "heritage":      {"type": "int", "range": [500_000_000, 5_000_000_000], "step": 100_000_000},
                "port_revenue":  {"type": "int", "range": [200_000_000, 2_000_000_000], "step": 100_000_000},
                "properties":    {"type": "int", "range": [5_000, 30_000], "step": 1_000},
                "elevation":     {"type": "choice", "values": [1.5, 2.0, 2.5, 3.0]},
                "wall_cost":     {"type": "int", "range": [500_000_000, 3_000_000_000], "step": 100_000_000},
                "retreat_cost":  {"type": "int", "range": [1_000_000_000, 5_000_000_000], "step": 200_000_000},
                "nature_cost":   {"type": "int", "range": [100_000_000, 800_000_000], "step": 50_000_000},
            },
            "answer_fn": None,
            "answer_template": (
                "UNDERDETERMINED: Each strategy involves different trade-offs between cost, "
                "cultural preservation, safety, and long-term risk. The optimal choice depends "
                "on community values and risk tolerance not determinable from data alone."
            ),
        },
    ],
}
