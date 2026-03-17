"""
Jurisdiction database for regulatory AI frameworks.

Contains 10 real-world regulatory frameworks and their representative
obligations. All data is based on publicly available regulatory texts.
No external dependencies — stdlib only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Framework:
    """A regulatory framework governing AI systems."""

    id: str
    name: str
    framework_type: str  # "binding", "voluntary", "hybrid"
    region: str
    enforcement_date: str  # ISO date
    penalty_min: float  # EUR
    penalty_max: float  # EUR
    penalty_basis: str  # "global_annual_turnover", "fixed", "per_violation"
    article_count: int
    categories: list
    status: str  # "enacted", "proposed", "voluntary"
    url: str
    description: str


@dataclass
class Obligation:
    """A single regulatory obligation within a framework."""

    id: str
    framework_id: str
    article: str
    title: str
    description: str
    obligation_type: str  # "mandatory", "recommended", "optional", "conditional"
    risk_level: Optional[str]
    category: str
    deadline: Optional[str]  # ISO date
    phase: Optional[str]


class JurisdictionDB:
    """In-memory database of regulatory frameworks and obligations."""

    def __init__(self) -> None:
        self.frameworks: dict[str, Framework] = self._init_frameworks()
        self.obligations: list[Obligation] = self._init_obligations()

    # ------------------------------------------------------------------
    # Framework definitions
    # ------------------------------------------------------------------

    def _init_frameworks(self) -> dict[str, Framework]:
        items = [
            Framework(
                id="eu_ai_act",
                name="EU Artificial Intelligence Act",
                framework_type="binding",
                region="EU",
                enforcement_date="2024-08-01",
                penalty_min=7_500_000.0,
                penalty_max=35_000_000.0,
                penalty_basis="global_annual_turnover",
                article_count=85,
                categories=[
                    "risk_classification",
                    "conformity_assessment",
                    "transparency",
                    "human_oversight",
                    "data_governance",
                    "technical_documentation",
                    "post_market_monitoring",
                    "registration",
                ],
                status="enacted",
                url="https://eur-lex.europa.eu/eli/reg/2024/1689/oj",
                description=(
                    "Regulation (EU) 2024/1689 laying down harmonised rules on "
                    "artificial intelligence. Establishes a risk-based classification "
                    "system with prohibitions on unacceptable-risk AI, strict "
                    "requirements for high-risk AI systems, and transparency "
                    "obligations for limited-risk AI."
                ),
            ),
            Framework(
                id="nist_ai_rmf",
                name="NIST AI Risk Management Framework",
                framework_type="voluntary",
                region="US",
                enforcement_date="2023-01-26",
                penalty_min=0.0,
                penalty_max=0.0,
                penalty_basis="fixed",
                article_count=72,
                categories=[
                    "govern",
                    "map",
                    "measure",
                    "manage",
                ],
                status="voluntary",
                url="https://www.nist.gov/artificial-intelligence/executive-order-safe-secure-and-trustworthy-artificial-intelligence",
                description=(
                    "Voluntary framework from the National Institute of Standards "
                    "and Technology structured around four core functions — Govern, "
                    "Map, Measure, and Manage — with 72 subcategories providing "
                    "actionable guidance for AI risk management."
                ),
            ),
            Framework(
                id="iso_42001",
                name="ISO/IEC 42001 AI Management System",
                framework_type="voluntary",
                region="Global",
                enforcement_date="2023-12-18",
                penalty_min=0.0,
                penalty_max=0.0,
                penalty_basis="fixed",
                article_count=39,
                categories=[
                    "leadership",
                    "planning",
                    "support",
                    "operation",
                    "performance_evaluation",
                    "improvement",
                ],
                status="voluntary",
                url="https://www.iso.org/standard/81230.html",
                description=(
                    "International standard specifying requirements for establishing, "
                    "implementing, maintaining, and continually improving an AI "
                    "management system. Provides a certification-based approach to "
                    "responsible AI governance."
                ),
            ),
            Framework(
                id="china_genai",
                name="China Interim Measures for Generative AI",
                framework_type="binding",
                region="China",
                enforcement_date="2023-08-15",
                penalty_min=10_000.0,
                penalty_max=100_000.0,
                penalty_basis="per_violation",
                article_count=24,
                categories=[
                    "content_safety",
                    "data_training",
                    "algorithmic_transparency",
                    "user_protection",
                    "security_assessment",
                ],
                status="enacted",
                url="http://www.cac.gov.cn/2023-07/13/c_1690898327029107.htm",
                description=(
                    "Interim administrative measures for the management of generative "
                    "AI services, issued by the Cyberspace Administration of China. "
                    "Requires security assessments, content labelling, and algorithmic "
                    "filing for generative AI service providers operating in China."
                ),
            ),
            Framework(
                id="gdpr_ai",
                name="EU General Data Protection Regulation (AI Provisions)",
                framework_type="binding",
                region="EU",
                enforcement_date="2018-05-25",
                penalty_min=10_000_000.0,
                penalty_max=20_000_000.0,
                penalty_basis="global_annual_turnover",
                article_count=15,
                categories=[
                    "automated_decision_making",
                    "data_protection_impact",
                    "lawful_basis",
                    "data_minimization",
                    "transparency",
                    "data_subject_rights",
                ],
                status="enacted",
                url="https://eur-lex.europa.eu/eli/reg/2016/679/oj",
                description=(
                    "Regulation (EU) 2016/679 on the protection of natural persons "
                    "with regard to the processing of personal data. Articles 5, 6, "
                    "13-14, 22, and 35-36 are directly relevant to AI systems that "
                    "process personal data or make automated decisions."
                ),
            ),
            Framework(
                id="sg_aiga",
                name="Singapore AI Governance Framework (Model Framework)",
                framework_type="voluntary",
                region="Singapore",
                enforcement_date="2020-01-20",
                penalty_min=0.0,
                penalty_max=0.0,
                penalty_basis="fixed",
                article_count=20,
                categories=[
                    "internal_governance",
                    "human_involvement",
                    "operations_management",
                    "stakeholder_engagement",
                ],
                status="voluntary",
                url="https://www.pdpc.gov.sg/help-and-resources/2020/01/model-ai-governance-framework",
                description=(
                    "Second edition of the Model AI Governance Framework published "
                    "by the Infocomm Media Development Authority of Singapore. "
                    "Provides practical and implementable guidance on ethical AI "
                    "deployment, organised around four guiding principles."
                ),
            ),
            Framework(
                id="uk_ai_framework",
                name="UK Pro-innovation AI Regulation Framework",
                framework_type="voluntary",
                region="UK",
                enforcement_date="2023-03-29",
                penalty_min=0.0,
                penalty_max=0.0,
                penalty_basis="fixed",
                article_count=25,
                categories=[
                    "safety",
                    "transparency",
                    "fairness",
                    "accountability",
                    "contestability",
                ],
                status="voluntary",
                url="https://www.gov.uk/government/publications/ai-regulation-a-pro-innovation-approach",
                description=(
                    "White paper setting out the UK government's principles-based "
                    "approach to AI regulation, delivered through existing sector "
                    "regulators rather than a new central AI authority. Built on "
                    "five cross-cutting principles."
                ),
            ),
            Framework(
                id="kr_ai_act",
                name="South Korea AI Basic Act",
                framework_type="binding",
                region="South Korea",
                enforcement_date="2025-01-21",
                penalty_min=0.0,
                penalty_max=30_000_000.0,
                penalty_basis="fixed",
                article_count=40,
                categories=[
                    "high_impact_ai",
                    "transparency",
                    "safety",
                    "user_rights",
                    "ai_ethics",
                ],
                status="proposed",
                url="https://www.law.go.kr",
                description=(
                    "Proposed comprehensive AI legislation in South Korea establishing "
                    "a framework for the development and use of artificial intelligence. "
                    "Introduces the concept of 'high-impact AI' requiring impact "
                    "assessments, transparency obligations, and user notification."
                ),
            ),
            Framework(
                id="ca_aida",
                name="Canada Artificial Intelligence and Data Act",
                framework_type="binding",
                region="Canada",
                enforcement_date="2025-01-01",
                penalty_min=0.0,
                penalty_max=25_000_000.0,
                penalty_basis="fixed",
                article_count=30,
                categories=[
                    "high_impact_systems",
                    "risk_assessment",
                    "transparency",
                    "record_keeping",
                    "accountability",
                ],
                status="proposed",
                url="https://www.parl.ca/legisinfo/en/bill/44-1/c-27",
                description=(
                    "Part 3 of Bill C-27 (Digital Charter Implementation Act, 2022). "
                    "Establishes requirements for responsible design, development, and "
                    "deployment of AI systems in Canada, with a focus on high-impact "
                    "systems and criminal prohibitions for reckless or malicious use."
                ),
            ),
            Framework(
                id="br_lgpd_ai",
                name="Brazil LGPD (AI-Relevant Provisions)",
                framework_type="binding",
                region="Brazil",
                enforcement_date="2020-09-18",
                penalty_min=0.0,
                penalty_max=50_000_000.0,
                penalty_basis="per_violation",
                article_count=10,
                categories=[
                    "automated_decision_making",
                    "data_protection",
                    "transparency",
                    "review_rights",
                ],
                status="enacted",
                url="https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm",
                description=(
                    "Lei Geral de Proteção de Dados Pessoais (Law No. 13,709/2018). "
                    "Article 20 grants data subjects the right to request review of "
                    "decisions made solely by automated processing. Penalties up to "
                    "2% of revenue in Brazil, capped at 50 million BRL per violation."
                ),
            ),
        ]
        return {fw.id: fw for fw in items}

    # ------------------------------------------------------------------
    # Obligation definitions
    # ------------------------------------------------------------------

    def _init_obligations(self) -> list[Obligation]:
        obs: list[Obligation] = []

        # ---- EU AI Act ----
        obs.extend([
            Obligation(
                id="euaia_risk_classification",
                framework_id="eu_ai_act",
                article="Art. 6",
                title="Risk classification of AI systems",
                description=(
                    "Providers must classify AI systems according to the risk "
                    "categories defined in Annex III. High-risk systems are subject "
                    "to mandatory requirements before being placed on the market."
                ),
                obligation_type="mandatory",
                risk_level="high",
                category="risk_classification",
                deadline="2025-08-02",
                phase="pre_market",
            ),
            Obligation(
                id="euaia_conformity_assessment",
                framework_id="eu_ai_act",
                article="Art. 43",
                title="Conformity assessment for high-risk AI",
                description=(
                    "High-risk AI systems must undergo conformity assessment "
                    "procedures before being placed on the market or put into "
                    "service. Certain biometric systems require third-party "
                    "assessment by a notified body."
                ),
                obligation_type="mandatory",
                risk_level="high",
                category="conformity_assessment",
                deadline="2025-08-02",
                phase="pre_market",
            ),
            Obligation(
                id="euaia_technical_documentation",
                framework_id="eu_ai_act",
                article="Art. 11",
                title="Technical documentation requirements",
                description=(
                    "Providers of high-risk AI systems must draw up technical "
                    "documentation demonstrating compliance with Chapter 2 "
                    "requirements. Documentation shall be kept up to date and "
                    "made available to national competent authorities upon request."
                ),
                obligation_type="mandatory",
                risk_level="high",
                category="technical_documentation",
                deadline="2025-08-02",
                phase="pre_market",
            ),
            Obligation(
                id="euaia_transparency",
                framework_id="eu_ai_act",
                article="Art. 13",
                title="Transparency and information to deployers",
                description=(
                    "High-risk AI systems must be designed to ensure their operation "
                    "is sufficiently transparent to enable deployers to interpret "
                    "the system's output and use it appropriately."
                ),
                obligation_type="mandatory",
                risk_level="high",
                category="transparency",
                deadline="2025-08-02",
                phase="pre_market",
            ),
            Obligation(
                id="euaia_transparency_limited",
                framework_id="eu_ai_act",
                article="Art. 52",
                title="Transparency for limited-risk AI (chatbots, deepfakes)",
                description=(
                    "Providers must ensure AI systems intended to interact with "
                    "persons are designed so that persons are informed they are "
                    "interacting with an AI system. Deep-fake content must be "
                    "disclosed as artificially generated or manipulated."
                ),
                obligation_type="mandatory",
                risk_level="limited",
                category="transparency",
                deadline="2025-02-02",
                phase="deployment",
            ),
            Obligation(
                id="euaia_human_oversight",
                framework_id="eu_ai_act",
                article="Art. 14",
                title="Human oversight measures",
                description=(
                    "High-risk AI systems shall be designed to be effectively "
                    "overseen by natural persons during use. Oversight measures "
                    "shall be commensurate with risks and the level of autonomy."
                ),
                obligation_type="mandatory",
                risk_level="high",
                category="human_oversight",
                deadline="2025-08-02",
                phase="deployment",
            ),
            Obligation(
                id="euaia_data_governance",
                framework_id="eu_ai_act",
                article="Art. 10",
                title="Data and data governance for training",
                description=(
                    "Training, validation, and testing data sets shall be subject "
                    "to appropriate data governance and management practices "
                    "concerning data collection, relevance, representativeness, "
                    "and bias examination."
                ),
                obligation_type="mandatory",
                risk_level="high",
                category="data_governance",
                deadline="2025-08-02",
                phase="development",
            ),
            Obligation(
                id="euaia_registration",
                framework_id="eu_ai_act",
                article="Art. 51",
                title="Registration in the EU database",
                description=(
                    "Providers of high-risk AI systems must register their systems "
                    "in the EU database established under Article 71 before placing "
                    "them on the market or putting them into service."
                ),
                obligation_type="mandatory",
                risk_level="high",
                category="registration",
                deadline="2025-08-02",
                phase="pre_market",
            ),
        ])

        # ---- NIST AI RMF ----
        obs.extend([
            Obligation(
                id="nist_govern_1_1",
                framework_id="nist_ai_rmf",
                article="GOVERN 1.1",
                title="Legal and regulatory requirements are identified",
                description=(
                    "Policies, processes, procedures, and practices are in place "
                    "to map, measure, and manage AI risks and are integrated into "
                    "broader enterprise risk management. Legal and regulatory "
                    "requirements involving AI are understood."
                ),
                obligation_type="recommended",
                risk_level=None,
                category="govern",
                deadline=None,
                phase="governance",
            ),
            Obligation(
                id="nist_govern_1_2",
                framework_id="nist_ai_rmf",
                article="GOVERN 1.2",
                title="Trustworthy AI characteristics are integrated",
                description=(
                    "The characteristics of trustworthy AI — valid and reliable, "
                    "safe, secure and resilient, accountable and transparent, "
                    "explainable and interpretable, privacy-enhanced, and fair "
                    "with harmful biases managed — are integrated into policies."
                ),
                obligation_type="recommended",
                risk_level=None,
                category="govern",
                deadline=None,
                phase="governance",
            ),
            Obligation(
                id="nist_govern_1_5",
                framework_id="nist_ai_rmf",
                article="GOVERN 1.5",
                title="Ongoing monitoring of AI risks",
                description=(
                    "Ongoing monitoring processes are in place to regularly assess "
                    "AI risks in light of changes to the AI system, its context "
                    "of use, or the broader environment."
                ),
                obligation_type="recommended",
                risk_level=None,
                category="govern",
                deadline=None,
                phase="governance",
            ),
            Obligation(
                id="nist_map_1_1",
                framework_id="nist_ai_rmf",
                article="MAP 1.1",
                title="Intended purpose and context of use are defined",
                description=(
                    "The intended purpose, potentially beneficial uses, context "
                    "of use, and the known limitations of the AI system are "
                    "documented. Assumptions and environmental factors are "
                    "identified and recorded."
                ),
                obligation_type="recommended",
                risk_level=None,
                category="map",
                deadline=None,
                phase="design",
            ),
            Obligation(
                id="nist_map_1_5",
                framework_id="nist_ai_rmf",
                article="MAP 1.5",
                title="Impacts to individuals and communities are identified",
                description=(
                    "Potential impacts to individuals, groups, communities, "
                    "organizations, and society are identified and documented. "
                    "Impacts are assessed across demographic groups, including "
                    "those historically underserved."
                ),
                obligation_type="recommended",
                risk_level=None,
                category="map",
                deadline=None,
                phase="design",
            ),
            Obligation(
                id="nist_measure_1_1",
                framework_id="nist_ai_rmf",
                article="MEASURE 1.1",
                title="Approaches for measurement of AI risks are established",
                description=(
                    "Appropriate methods and metrics are identified and applied "
                    "to measure AI risks and trustworthiness characteristics. "
                    "Measurement approaches are aligned with the system's context."
                ),
                obligation_type="recommended",
                risk_level=None,
                category="measure",
                deadline=None,
                phase="evaluation",
            ),
            Obligation(
                id="nist_measure_2_6",
                framework_id="nist_ai_rmf",
                article="MEASURE 2.6",
                title="AI system performance measured for fairness",
                description=(
                    "Computational tests for bias are conducted and results are "
                    "evaluated. Disparate impact metrics are computed across "
                    "demographic groups for relevant performance measures."
                ),
                obligation_type="recommended",
                risk_level=None,
                category="measure",
                deadline=None,
                phase="evaluation",
            ),
            Obligation(
                id="nist_manage_1_1",
                framework_id="nist_ai_rmf",
                article="MANAGE 1.1",
                title="AI risk treatment plans are developed",
                description=(
                    "A determination is made as to whether the AI system achieves "
                    "its intended purpose and whether its benefits outweigh its "
                    "risks. Risk treatment actions are prioritised and resources "
                    "are allocated."
                ),
                obligation_type="recommended",
                risk_level=None,
                category="manage",
                deadline=None,
                phase="deployment",
            ),
        ])

        # ---- ISO 42001 ----
        obs.extend([
            Obligation(
                id="iso42001_leadership",
                framework_id="iso_42001",
                article="Clause 5",
                title="Leadership and commitment",
                description=(
                    "Top management shall demonstrate leadership and commitment "
                    "with respect to the AI management system by ensuring the AI "
                    "policy and objectives are established and compatible with the "
                    "strategic direction of the organization."
                ),
                obligation_type="recommended",
                risk_level=None,
                category="leadership",
                deadline=None,
                phase="governance",
            ),
            Obligation(
                id="iso42001_risk_assessment",
                framework_id="iso_42001",
                article="Clause 6.1",
                title="Actions to address risks and opportunities",
                description=(
                    "The organization shall determine risks and opportunities "
                    "that need to be addressed to give assurance that the AI "
                    "management system can achieve its intended outcomes, prevent "
                    "or reduce undesired effects, and achieve continual improvement."
                ),
                obligation_type="recommended",
                risk_level=None,
                category="planning",
                deadline=None,
                phase="planning",
            ),
            Obligation(
                id="iso42001_competence",
                framework_id="iso_42001",
                article="Clause 7.2",
                title="Competence of personnel",
                description=(
                    "The organization shall determine the necessary competence "
                    "of persons doing work under its control that affects AI "
                    "system performance and the effectiveness of the AIMS, "
                    "and ensure these persons are competent."
                ),
                obligation_type="recommended",
                risk_level=None,
                category="support",
                deadline=None,
                phase="governance",
            ),
            Obligation(
                id="iso42001_ai_impact",
                framework_id="iso_42001",
                article="Clause 6.1.4",
                title="AI system impact assessment",
                description=(
                    "The organization shall conduct an impact assessment for each "
                    "AI system within scope, considering impacts on individuals, "
                    "groups, and society. The assessment shall address potential "
                    "harms and benefits."
                ),
                obligation_type="recommended",
                risk_level=None,
                category="planning",
                deadline=None,
                phase="planning",
            ),
            Obligation(
                id="iso42001_operational_planning",
                framework_id="iso_42001",
                article="Clause 8.1",
                title="Operational planning and control",
                description=(
                    "The organization shall plan, implement, and control the "
                    "processes needed to meet AI management system requirements "
                    "by establishing criteria for the processes and implementing "
                    "control of the processes in accordance with the criteria."
                ),
                obligation_type="recommended",
                risk_level=None,
                category="operation",
                deadline=None,
                phase="deployment",
            ),
            Obligation(
                id="iso42001_monitoring",
                framework_id="iso_42001",
                article="Clause 9.1",
                title="Monitoring, measurement, analysis, and evaluation",
                description=(
                    "The organization shall determine what needs to be monitored "
                    "and measured, including AI system performance and the "
                    "effectiveness of the AIMS. Methods for monitoring and "
                    "analysis shall be determined."
                ),
                obligation_type="recommended",
                risk_level=None,
                category="performance_evaluation",
                deadline=None,
                phase="monitoring",
            ),
        ])

        # ---- China GenAI ----
        obs.extend([
            Obligation(
                id="china_security_assessment",
                framework_id="china_genai",
                article="Art. 17",
                title="Security assessment before public launch",
                description=(
                    "Providers of generative AI services with public opinion "
                    "properties or social mobilisation capability must submit a "
                    "security assessment to the Cyberspace Administration of China "
                    "and complete algorithm filing before providing services."
                ),
                obligation_type="mandatory",
                risk_level="high",
                category="security_assessment",
                deadline=None,
                phase="pre_market",
            ),
            Obligation(
                id="china_training_data",
                framework_id="china_genai",
                article="Art. 7",
                title="Training data lawfulness",
                description=(
                    "Providers shall use data from lawful sources for training. "
                    "Intellectual property rights must be respected. Personal "
                    "information handling must comply with relevant laws. Measures "
                    "shall improve the quality of training data."
                ),
                obligation_type="mandatory",
                risk_level=None,
                category="data_training",
                deadline=None,
                phase="development",
            ),
            Obligation(
                id="china_content_labelling",
                framework_id="china_genai",
                article="Art. 12",
                title="Content labelling and watermarking",
                description=(
                    "Generated content including images, videos, and audio must "
                    "be labelled in accordance with relevant regulations. Providers "
                    "must add watermarks to generated content to ensure traceability."
                ),
                obligation_type="mandatory",
                risk_level=None,
                category="content_safety",
                deadline=None,
                phase="deployment",
            ),
            Obligation(
                id="china_user_protection",
                framework_id="china_genai",
                article="Art. 9",
                title="User rights and complaint mechanisms",
                description=(
                    "Providers shall establish complaint and reporting mechanisms. "
                    "Users must be able to report illegal content. Providers must "
                    "handle complaints promptly and publish complaint channels."
                ),
                obligation_type="mandatory",
                risk_level=None,
                category="user_protection",
                deadline=None,
                phase="deployment",
            ),
            Obligation(
                id="china_algo_transparency",
                framework_id="china_genai",
                article="Art. 17",
                title="Algorithm registration and filing",
                description=(
                    "Providers must complete algorithm filing procedures with the "
                    "Cyberspace Administration of China, disclosing algorithm "
                    "properties, intended use, and self-assessment results."
                ),
                obligation_type="mandatory",
                risk_level=None,
                category="algorithmic_transparency",
                deadline=None,
                phase="pre_market",
            ),
            Obligation(
                id="china_content_review",
                framework_id="china_genai",
                article="Art. 4",
                title="Core socialist values in generated content",
                description=(
                    "Generated content must adhere to core socialist values. "
                    "Content must not endanger national security, incite "
                    "subversion, or undermine national unity. Providers are "
                    "responsible for the legality of generated content."
                ),
                obligation_type="mandatory",
                risk_level=None,
                category="content_safety",
                deadline=None,
                phase="deployment",
            ),
        ])

        # ---- GDPR AI provisions ----
        obs.extend([
            Obligation(
                id="gdpr_dpia",
                framework_id="gdpr_ai",
                article="Art. 35",
                title="Data Protection Impact Assessment",
                description=(
                    "Where processing using new technologies is likely to result "
                    "in a high risk to the rights and freedoms of natural persons, "
                    "the controller shall carry out an assessment of the impact of "
                    "the envisaged processing operations on the protection of "
                    "personal data, including AI-based profiling and scoring."
                ),
                obligation_type="mandatory",
                risk_level="high",
                category="data_protection_impact",
                deadline=None,
                phase="pre_market",
            ),
            Obligation(
                id="gdpr_lawful_basis",
                framework_id="gdpr_ai",
                article="Art. 6",
                title="Lawful basis for processing",
                description=(
                    "Processing of personal data for AI training or inference "
                    "shall be lawful only if at least one of the six lawful bases "
                    "applies: consent, contract, legal obligation, vital interests, "
                    "public task, or legitimate interests."
                ),
                obligation_type="mandatory",
                risk_level=None,
                category="lawful_basis",
                deadline=None,
                phase="development",
            ),
            Obligation(
                id="gdpr_automated_decisions",
                framework_id="gdpr_ai",
                article="Art. 22",
                title="Automated individual decision-making including profiling",
                description=(
                    "Data subjects have the right not to be subject to a decision "
                    "based solely on automated processing, including profiling, "
                    "which produces legal effects or similarly significantly affects "
                    "them. Suitable safeguards must be implemented."
                ),
                obligation_type="mandatory",
                risk_level="high",
                category="automated_decision_making",
                deadline=None,
                phase="deployment",
            ),
            Obligation(
                id="gdpr_data_minimization",
                framework_id="gdpr_ai",
                article="Art. 5(1)(c)",
                title="Data minimization principle",
                description=(
                    "Personal data shall be adequate, relevant, and limited to "
                    "what is necessary in relation to the purposes for which they "
                    "are processed. AI training data must not include unnecessary "
                    "personal data categories."
                ),
                obligation_type="mandatory",
                risk_level=None,
                category="data_minimization",
                deadline=None,
                phase="development",
            ),
            Obligation(
                id="gdpr_transparency_13",
                framework_id="gdpr_ai",
                article="Art. 13-14",
                title="Transparency and information to data subjects",
                description=(
                    "Controllers must provide data subjects with information about "
                    "the existence of automated decision-making, including profiling, "
                    "meaningful information about the logic involved, and the "
                    "significance and envisaged consequences of such processing."
                ),
                obligation_type="mandatory",
                risk_level=None,
                category="transparency",
                deadline=None,
                phase="deployment",
            ),
            Obligation(
                id="gdpr_right_to_explanation",
                framework_id="gdpr_ai",
                article="Recital 71",
                title="Right to explanation of automated decisions",
                description=(
                    "Data subjects should have the right to obtain an explanation "
                    "of the decision reached after automated assessment and to "
                    "challenge the decision. Suitable safeguards should include "
                    "the right to obtain human intervention."
                ),
                obligation_type="mandatory",
                risk_level=None,
                category="data_subject_rights",
                deadline=None,
                phase="deployment",
            ),
        ])

        # ---- Singapore AIGA ----
        obs.extend([
            Obligation(
                id="sg_internal_governance",
                framework_id="sg_aiga",
                article="Section 2.1",
                title="Internal governance structures and measures",
                description=(
                    "Organizations should establish governance structures for "
                    "clear roles and responsibilities for ethical AI deployment. "
                    "Risk management and internal controls should be adapted to "
                    "account for AI-related risks."
                ),
                obligation_type="recommended",
                risk_level=None,
                category="internal_governance",
                deadline=None,
                phase="governance",
            ),
            Obligation(
                id="sg_risk_aware",
                framework_id="sg_aiga",
                article="Section 2.2",
                title="Determining AI decision-making level of human involvement",
                description=(
                    "Organizations should implement a risk assessment approach "
                    "to determine the appropriate level of human involvement in "
                    "AI-augmented decision-making. The severity and probability "
                    "of harm should guide the level of oversight."
                ),
                obligation_type="recommended",
                risk_level=None,
                category="human_involvement",
                deadline=None,
                phase="design",
            ),
            Obligation(
                id="sg_operations_management",
                framework_id="sg_aiga",
                article="Section 3.1",
                title="Minimizing bias in data and models",
                description=(
                    "Organizations should implement measures to minimize bias "
                    "in AI systems through careful data collection, preparation, "
                    "and model validation. Regular bias audits should be conducted "
                    "to detect and address unfair outcomes."
                ),
                obligation_type="recommended",
                risk_level=None,
                category="operations_management",
                deadline=None,
                phase="development",
            ),
            Obligation(
                id="sg_explainability",
                framework_id="sg_aiga",
                article="Section 3.2",
                title="Providing for algorithmic explainability",
                description=(
                    "Organizations should provide the appropriate level of "
                    "explanation of AI decisions based on the impact and "
                    "context. Stakeholders should be able to understand how "
                    "decisions that affect them are made."
                ),
                obligation_type="recommended",
                risk_level=None,
                category="operations_management",
                deadline=None,
                phase="deployment",
            ),
            Obligation(
                id="sg_stakeholder",
                framework_id="sg_aiga",
                article="Section 4",
                title="Stakeholder interaction and communication",
                description=(
                    "Organizations should regularly communicate with stakeholders "
                    "about their AI systems, including their capabilities and "
                    "limitations. Feedback mechanisms should be established."
                ),
                obligation_type="recommended",
                risk_level=None,
                category="stakeholder_engagement",
                deadline=None,
                phase="deployment",
            ),
        ])

        # ---- UK AI Framework ----
        obs.extend([
            Obligation(
                id="uk_safety",
                framework_id="uk_ai_framework",
                article="Principle 1",
                title="Safety, security, and robustness",
                description=(
                    "AI systems should function in a robust, secure, and safe "
                    "way throughout their life cycle, and risks should be "
                    "continually identified, assessed, and managed."
                ),
                obligation_type="recommended",
                risk_level=None,
                category="safety",
                deadline=None,
                phase="lifecycle",
            ),
            Obligation(
                id="uk_transparency",
                framework_id="uk_ai_framework",
                article="Principle 2",
                title="Appropriate transparency and explainability",
                description=(
                    "AI systems should be appropriately transparent and "
                    "explainable. The degree of transparency should be "
                    "proportionate to the level of risk and context."
                ),
                obligation_type="recommended",
                risk_level=None,
                category="transparency",
                deadline=None,
                phase="deployment",
            ),
            Obligation(
                id="uk_fairness",
                framework_id="uk_ai_framework",
                article="Principle 3",
                title="Fairness",
                description=(
                    "AI systems should not undermine the legal rights of "
                    "individuals or organizations, create unfair market "
                    "outcomes, or discriminate against individuals or groups."
                ),
                obligation_type="recommended",
                risk_level=None,
                category="fairness",
                deadline=None,
                phase="lifecycle",
            ),
            Obligation(
                id="uk_accountability",
                framework_id="uk_ai_framework",
                article="Principle 4",
                title="Accountability and governance",
                description=(
                    "Appropriate governance measures should be in place to "
                    "ensure effective oversight of AI systems. Clear lines of "
                    "accountability should be established and maintained."
                ),
                obligation_type="recommended",
                risk_level=None,
                category="accountability",
                deadline=None,
                phase="governance",
            ),
            Obligation(
                id="uk_contestability",
                framework_id="uk_ai_framework",
                article="Principle 5",
                title="Contestability and redress",
                description=(
                    "Users and affected third parties should be able to contest "
                    "harmful outcomes or decisions generated by AI systems. "
                    "Clear routes to redress should be available."
                ),
                obligation_type="recommended",
                risk_level=None,
                category="contestability",
                deadline=None,
                phase="deployment",
            ),
        ])

        # ---- South Korea AI Act ----
        obs.extend([
            Obligation(
                id="kr_high_impact",
                framework_id="kr_ai_act",
                article="Art. 27",
                title="High-impact AI designation and assessment",
                description=(
                    "AI systems that significantly affect life, physical safety, "
                    "or fundamental rights shall be designated as high-impact AI. "
                    "Operators of high-impact AI must conduct impact assessments "
                    "before deployment."
                ),
                obligation_type="mandatory",
                risk_level="high",
                category="high_impact_ai",
                deadline=None,
                phase="pre_market",
            ),
            Obligation(
                id="kr_transparency_notice",
                framework_id="kr_ai_act",
                article="Art. 22",
                title="Transparency and user notification",
                description=(
                    "Users must be notified when interacting with an AI system. "
                    "The fact that content has been generated or decisions have "
                    "been made by AI must be clearly disclosed to affected persons."
                ),
                obligation_type="mandatory",
                risk_level=None,
                category="transparency",
                deadline=None,
                phase="deployment",
            ),
            Obligation(
                id="kr_safety_measures",
                framework_id="kr_ai_act",
                article="Art. 25",
                title="Safety measures for AI systems",
                description=(
                    "Developers and operators must implement appropriate safety "
                    "measures, including testing, monitoring, and incident "
                    "response procedures, to prevent harm from AI systems."
                ),
                obligation_type="mandatory",
                risk_level=None,
                category="safety",
                deadline=None,
                phase="lifecycle",
            ),
            Obligation(
                id="kr_ethics_principles",
                framework_id="kr_ai_act",
                article="Art. 5",
                title="AI ethics principles",
                description=(
                    "AI development and use should respect human dignity, "
                    "promote public good, prevent discrimination, ensure "
                    "transparency, and maintain accountability."
                ),
                obligation_type="mandatory",
                risk_level=None,
                category="ai_ethics",
                deadline=None,
                phase="lifecycle",
            ),
            Obligation(
                id="kr_user_rights",
                framework_id="kr_ai_act",
                article="Art. 30",
                title="Rights of users affected by AI decisions",
                description=(
                    "Users affected by high-impact AI decisions have the right "
                    "to request an explanation of the decision, refuse solely "
                    "automated decisions, and seek remedy for damage caused."
                ),
                obligation_type="mandatory",
                risk_level="high",
                category="user_rights",
                deadline=None,
                phase="deployment",
            ),
        ])

        # ---- Canada AIDA ----
        obs.extend([
            Obligation(
                id="ca_high_impact_assessment",
                framework_id="ca_aida",
                article="s. 7",
                title="Assessment of high-impact AI systems",
                description=(
                    "Persons responsible for high-impact AI systems must assess "
                    "whether the system is a high-impact system based on criteria "
                    "established by regulation, considering the context and "
                    "potential adverse impacts."
                ),
                obligation_type="mandatory",
                risk_level="high",
                category="high_impact_systems",
                deadline=None,
                phase="pre_market",
            ),
            Obligation(
                id="ca_risk_mitigation",
                framework_id="ca_aida",
                article="s. 8",
                title="Risk mitigation measures",
                description=(
                    "Persons responsible for high-impact AI systems must establish "
                    "measures to identify, assess, and mitigate risks of harm or "
                    "biased output that could result from the use of the system."
                ),
                obligation_type="mandatory",
                risk_level="high",
                category="risk_assessment",
                deadline=None,
                phase="development",
            ),
            Obligation(
                id="ca_transparency_general",
                framework_id="ca_aida",
                article="s. 11",
                title="General transparency duty",
                description=(
                    "Persons who make available a high-impact AI system must "
                    "publish a plain-language description of the system, "
                    "including how it is used, the type of content it generates, "
                    "decisions it makes, and the mitigation measures in place."
                ),
                obligation_type="mandatory",
                risk_level="high",
                category="transparency",
                deadline=None,
                phase="deployment",
            ),
            Obligation(
                id="ca_record_keeping",
                framework_id="ca_aida",
                article="s. 12",
                title="Record keeping requirements",
                description=(
                    "Persons responsible for high-impact AI systems must keep "
                    "records of their compliance with obligations under the Act, "
                    "including records of risk assessments and mitigation measures."
                ),
                obligation_type="mandatory",
                risk_level="high",
                category="record_keeping",
                deadline=None,
                phase="lifecycle",
            ),
            Obligation(
                id="ca_notification",
                framework_id="ca_aida",
                article="s. 12(2)",
                title="Notification of material harm",
                description=(
                    "When a high-impact AI system causes or is likely to cause "
                    "material harm, the person responsible must notify the "
                    "AI and Data Commissioner as soon as feasible."
                ),
                obligation_type="mandatory",
                risk_level="high",
                category="accountability",
                deadline=None,
                phase="deployment",
            ),
            Obligation(
                id="ca_criminal_prohibition",
                framework_id="ca_aida",
                article="s. 39",
                title="Criminal prohibition on reckless AI deployment",
                description=(
                    "It is a criminal offence to deploy an AI system where the "
                    "person knows or is reckless as to whether its use could "
                    "cause serious physical or psychological harm, or "
                    "substantial damage to property."
                ),
                obligation_type="mandatory",
                risk_level="high",
                category="accountability",
                deadline=None,
                phase="deployment",
            ),
        ])

        # ---- Brazil LGPD AI ----
        obs.extend([
            Obligation(
                id="br_automated_review",
                framework_id="br_lgpd_ai",
                article="Art. 20",
                title="Right to review of automated decisions",
                description=(
                    "The data subject has the right to request the controller "
                    "to review decisions made solely based on automated "
                    "processing of personal data that affect his or her "
                    "interests, including decisions to define personal, "
                    "professional, consumer, or credit profiles."
                ),
                obligation_type="mandatory",
                risk_level=None,
                category="automated_decision_making",
                deadline=None,
                phase="deployment",
            ),
            Obligation(
                id="br_explanation_criteria",
                framework_id="br_lgpd_ai",
                article="Art. 20(1)",
                title="Explanation of automated decision criteria",
                description=(
                    "The controller shall provide clear and adequate information "
                    "regarding the criteria and procedures used for automated "
                    "decision-making, subject to trade-secret protections."
                ),
                obligation_type="mandatory",
                risk_level=None,
                category="transparency",
                deadline=None,
                phase="deployment",
            ),
            Obligation(
                id="br_anpd_audit",
                framework_id="br_lgpd_ai",
                article="Art. 20(2)",
                title="ANPD audit authority",
                description=(
                    "The national authority (ANPD) may carry out an audit to "
                    "verify discriminatory aspects of automated processing of "
                    "personal data when the controller does not provide "
                    "adequate information under trade-secret claims."
                ),
                obligation_type="conditional",
                risk_level=None,
                category="review_rights",
                deadline=None,
                phase="enforcement",
            ),
            Obligation(
                id="br_data_protection_officer",
                framework_id="br_lgpd_ai",
                article="Art. 41",
                title="Designation of a Data Protection Officer",
                description=(
                    "Controllers that carry out large-scale processing of "
                    "personal data, including through AI systems, must appoint "
                    "a data protection officer whose identity and contact "
                    "information is publicly available."
                ),
                obligation_type="mandatory",
                risk_level=None,
                category="data_protection",
                deadline=None,
                phase="governance",
            ),
            Obligation(
                id="br_impact_report",
                framework_id="br_lgpd_ai",
                article="Art. 38",
                title="Data protection impact report",
                description=(
                    "The ANPD may require the controller to prepare a data "
                    "protection impact report, including for automated "
                    "processing activities, describing the types of data "
                    "collected, methodology, and safeguards applied."
                ),
                obligation_type="conditional",
                risk_level=None,
                category="data_protection",
                deadline=None,
                phase="pre_market",
            ),
        ])

        return obs

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_framework(self, framework_id: str) -> Framework:
        """Return a single framework by id, or raise KeyError."""
        if framework_id not in self.frameworks:
            raise KeyError(f"Unknown framework: {framework_id}")
        return self.frameworks[framework_id]

    def get_frameworks(
        self,
        region: Optional[str] = None,
        framework_type: Optional[str] = None,
    ) -> list[Framework]:
        """Return frameworks, optionally filtered by region and/or type."""
        results = list(self.frameworks.values())
        if region is not None:
            results = [f for f in results if f.region == region]
        if framework_type is not None:
            results = [f for f in results if f.framework_type == framework_type]
        return results

    def get_obligations(
        self,
        framework_id: Optional[str] = None,
        category: Optional[str] = None,
        risk_level: Optional[str] = None,
        obligation_type: Optional[str] = None,
    ) -> list[Obligation]:
        """Return obligations matching all supplied filters."""
        results = list(self.obligations)
        if framework_id is not None:
            results = [o for o in results if o.framework_id == framework_id]
        if category is not None:
            results = [o for o in results if o.category == category]
        if risk_level is not None:
            results = [o for o in results if o.risk_level == risk_level]
        if obligation_type is not None:
            results = [o for o in results if o.obligation_type == obligation_type]
        return results

    def get_obligation(self, obligation_id: str) -> Obligation:
        """Return a single obligation by id, or raise KeyError."""
        for ob in self.obligations:
            if ob.id == obligation_id:
                return ob
        raise KeyError(f"Unknown obligation: {obligation_id}")

    def search_obligations(self, query: str) -> list[Obligation]:
        """Case-insensitive text search across obligation titles and descriptions."""
        q = query.lower()
        return [
            o
            for o in self.obligations
            if q in o.title.lower() or q in o.description.lower()
        ]

    def get_frameworks_by_date_range(
        self, start: str, end: str
    ) -> list[Framework]:
        """Return frameworks whose enforcement_date falls within [start, end]."""
        return [
            f
            for f in self.frameworks.values()
            if start <= f.enforcement_date <= end
        ]

    def get_binding_frameworks(self) -> list[Framework]:
        """Return all binding (legally enforceable) frameworks."""
        return self.get_frameworks(framework_type="binding")

    def get_voluntary_frameworks(self) -> list[Framework]:
        """Return all voluntary / guidance-based frameworks."""
        return self.get_frameworks(framework_type="voluntary")

    def get_categories(self) -> list[str]:
        """Return sorted unique categories across all frameworks."""
        cats: set[str] = set()
        for fw in self.frameworks.values():
            cats.update(fw.categories)
        return sorted(cats)

    def get_regions(self) -> list[str]:
        """Return sorted unique regions across all frameworks."""
        return sorted({f.region for f in self.frameworks.values()})

    def get_obligations_by_deadline(self, before: str) -> list[Obligation]:
        """Return obligations whose deadline is on or before the given date."""
        return [
            o
            for o in self.obligations
            if o.deadline is not None and o.deadline <= before
        ]

    def count_obligations(self, framework_id: Optional[str] = None) -> int:
        """Count obligations, optionally scoped to a framework."""
        if framework_id is not None:
            return sum(
                1 for o in self.obligations if o.framework_id == framework_id
            )
        return len(self.obligations)

    def summary(self) -> str:
        """Return a human-readable summary of the database contents."""
        fw_count = len(self.frameworks)
        ob_count = len(self.obligations)
        regions = ", ".join(self.get_regions())
        binding = len(self.get_binding_frameworks())
        voluntary = len(self.get_voluntary_frameworks())
        lines = [
            "Jurisdiction Database Summary",
            "=" * 40,
            f"Frameworks : {fw_count} ({binding} binding, {voluntary} voluntary)",
            f"Obligations: {ob_count}",
            f"Regions    : {regions}",
            "",
        ]
        for fw in self.frameworks.values():
            n_obs = self.count_obligations(fw.id)
            lines.append(
                f"  [{fw.id}] {fw.name} — {fw.framework_type}, "
                f"{fw.region}, {n_obs} obligations"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_json(self) -> dict:
        """Serialize the entire database to a JSON-compatible dict."""
        return {
            "frameworks": {
                fid: asdict(fw) for fid, fw in self.frameworks.items()
            },
            "obligations": [asdict(o) for o in self.obligations],
        }

    @classmethod
    def from_json(cls, data: dict) -> "JurisdictionDB":
        """Deserialize a database from a dict produced by to_json()."""
        db = cls.__new__(cls)
        db.frameworks = {
            fid: Framework(**fdata)
            for fid, fdata in data["frameworks"].items()
        }
        db.obligations = [Obligation(**odata) for odata in data["obligations"]]
        return db
