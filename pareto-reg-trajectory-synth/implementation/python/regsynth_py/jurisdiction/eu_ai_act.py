"""EU Artificial Intelligence Act detailed encoding.

Provides comprehensive encoding of the EU AI Act including risk categories,
article obligations, phase-in milestones, annex requirements, and penalty tiers.
"""

from dataclasses import dataclass, field
from enum import Enum
import json


class RiskCategory(Enum):
    """EU AI Act risk classification levels."""
    UNACCEPTABLE = "unacceptable"
    HIGH = "high"
    LIMITED = "limited"
    MINIMAL = "minimal"
    SYSTEMIC = "systemic"


@dataclass
class PhaseInMilestone:
    """A phase-in milestone for the EU AI Act."""
    date: str
    description: str
    articles: list
    requirements: list


@dataclass
class AnnexRequirement:
    """A requirement from the EU AI Act annexes."""
    annex: str
    title: str
    items: list


@dataclass
class ArticleObligation:
    """An obligation derived from an EU AI Act article."""
    article: str
    title: str
    summary: str
    risk_category: str
    obligation_type: str
    requirements: list
    deadline: str = None
    penalties: dict = None


class EUAIAct:
    """Comprehensive encoding of the EU Artificial Intelligence Act.

    Regulation (EU) 2024/1689 laying down harmonised rules on artificial
    intelligence, entered into force August 1, 2024 with phased application.
    """

    def __init__(self):
        self.risk_categories = self._init_risk_categories()
        self.milestones = self._init_phase_in_milestones()
        self.articles = self._init_articles()
        self.annexes = self._init_annexes()

    def _init_risk_categories(self) -> dict:
        """Define risk categories with examples from the Act."""
        return {
            RiskCategory.UNACCEPTABLE.value: {
                "level": "unacceptable",
                "description": "AI practices that are prohibited due to unacceptable risk",
                "legal_basis": "Title II, Article 5",
                "examples": [
                    "Social scoring by public authorities",
                    "Real-time remote biometric identification in publicly accessible spaces for law enforcement (with exceptions)",
                    "Subliminal manipulation causing harm",
                    "Exploitation of vulnerabilities of specific groups",
                    "Biometric categorisation inferring sensitive attributes",
                    "Untargeted scraping of facial images for facial recognition databases",
                    "Emotion recognition in workplace and education (with exceptions)",
                    "Individual predictive policing based solely on profiling",
                ],
                "consequence": "Prohibited — must not be placed on the market or used",
            },
            RiskCategory.HIGH.value: {
                "level": "high",
                "description": "AI systems posing significant risk to health, safety, or fundamental rights",
                "legal_basis": "Title III, Articles 6-51",
                "annex_iii_areas": [
                    "Biometrics (remote biometric identification, categorisation)",
                    "Critical infrastructure (safety components of road traffic, water/gas/electricity supply)",
                    "Education and vocational training (admission, assessment, proctoring)",
                    "Employment (recruitment, task allocation, performance monitoring, termination)",
                    "Access to essential services (credit scoring, emergency services, health/life insurance)",
                    "Law enforcement (individual risk assessment, polygraphs, evidence evaluation, profiling)",
                    "Migration, asylum and border control (polygraphs, risk assessment, document verification)",
                    "Administration of justice (legal research, applying law to facts)",
                ],
                "consequence": "Subject to conformity assessment, CE marking, and ongoing monitoring",
            },
            RiskCategory.LIMITED.value: {
                "level": "limited",
                "description": "AI systems with specific transparency obligations",
                "legal_basis": "Title IV, Article 50",
                "examples": [
                    "Chatbots and conversational AI (disclose AI interaction)",
                    "Emotion recognition systems (inform subjects)",
                    "Biometric categorisation systems (inform subjects)",
                    "Deep fakes / synthetic media (label as AI-generated)",
                    "AI-generated text published for public information (label as AI-generated)",
                ],
                "consequence": "Transparency obligations must be met",
            },
            RiskCategory.MINIMAL.value: {
                "level": "minimal",
                "description": "AI systems with minimal or no risk — no specific obligations",
                "legal_basis": "Not specifically regulated",
                "examples": [
                    "AI-enabled video games",
                    "Spam filters",
                    "Inventory management systems",
                    "AI-assisted manufacturing optimisation",
                ],
                "consequence": "No specific obligations; voluntary codes of conduct encouraged",
            },
            RiskCategory.SYSTEMIC.value: {
                "level": "systemic",
                "description": "General-purpose AI models with systemic risk",
                "legal_basis": "Chapter V, Articles 51-56",
                "threshold": "Trained with >10^25 FLOPs cumulative compute, or designated by Commission",
                "examples": [
                    "Large foundation models exceeding compute threshold",
                    "Models with high-impact capabilities designated by Commission decision",
                ],
                "consequence": "Additional obligations: model evaluation, adversarial testing, incident reporting, cybersecurity",
            },
        }

    def _init_phase_in_milestones(self) -> list:
        """Define the phased application timeline."""
        return [
            PhaseInMilestone(
                date="2025-02-02",
                description="Prohibited AI practices (Title II) and AI literacy (Article 4) take effect",
                articles=["Article 5", "Article 4"],
                requirements=[
                    "Cease all prohibited AI practices listed in Article 5",
                    "Ensure AI literacy among staff operating or overseeing AI systems",
                ],
            ),
            PhaseInMilestone(
                date="2025-08-02",
                description="GPAI model provisions, governance and confidentiality rules apply",
                articles=["Article 51", "Article 52", "Article 53", "Article 54", "Article 55", "Article 78", "Article 79"],
                requirements=[
                    "GPAI model providers must comply with transparency and documentation requirements",
                    "GPAI models with systemic risk: model evaluation, adversarial testing, incident tracking",
                    "AI Office operational and governance structure established",
                    "Codes of practice for GPAI finalised",
                ],
            ),
            PhaseInMilestone(
                date="2026-08-02",
                description="Most provisions apply including high-risk AI in Annex III",
                articles=[
                    "Article 6", "Article 9", "Article 10", "Article 11", "Article 12",
                    "Article 13", "Article 14", "Article 15", "Article 16", "Article 26",
                    "Article 43", "Article 47", "Article 49", "Article 50", "Article 51",
                    "Article 72", "Article 85",
                ],
                requirements=[
                    "High-risk AI systems (Annex III) must comply with all Chapter III requirements",
                    "Conformity assessments completed for high-risk systems",
                    "CE marking affixed to compliant high-risk systems",
                    "Registration in EU database completed",
                    "Post-market monitoring plans in place",
                    "Transparency obligations for limited-risk systems enforced",
                    "Penalties applicable for non-compliance",
                ],
            ),
            PhaseInMilestone(
                date="2027-08-02",
                description="High-risk AI systems in Annex I (EU harmonisation legislation) apply",
                articles=["Article 6(1)", "Annex I"],
                requirements=[
                    "AI systems that are safety components of products under EU harmonisation legislation must comply",
                    "Full integration with existing product safety frameworks",
                ],
            ),
        ]

    def _init_articles(self) -> list:
        """Define article obligations with real content."""
        return [
            ArticleObligation(
                article="Article 5",
                title="Prohibited artificial intelligence practices",
                summary="Prohibits AI systems that deploy subliminal manipulation, exploit vulnerabilities, perform social scoring, use real-time remote biometric identification for law enforcement (with narrow exceptions), infer emotions in workplaces/education, perform untargeted facial image scraping, or engage in biometric categorisation inferring sensitive attributes.",
                risk_category="unacceptable",
                obligation_type="mandatory",
                requirements=[
                    "Do not place on market, put into service, or use any AI system falling under prohibited practices",
                    "Identify and discontinue any existing prohibited AI uses",
                    "Document assessment of whether systems fall under prohibitions",
                ],
                deadline="2025-02-02",
                penalties={"max_eur": 35_000_000, "max_turnover_pct": 7.0},
            ),
            ArticleObligation(
                article="Article 6",
                title="Classification rules for high-risk AI systems",
                summary="Defines two routes to high-risk classification: (1) AI as safety component of product covered by Annex I legislation requiring third-party conformity assessment, (2) AI systems in Annex III areas unless output is purely accessory.",
                risk_category="high",
                obligation_type="mandatory",
                requirements=[
                    "Assess whether AI system falls under Annex I or Annex III",
                    "Document classification rationale",
                    "Apply exception criteria if system does not pose significant risk",
                ],
                deadline="2026-08-02",
            ),
            ArticleObligation(
                article="Article 9",
                title="Risk management system",
                summary="Providers of high-risk AI must establish, implement, document, and maintain a continuous iterative risk management system throughout the system lifecycle.",
                risk_category="high",
                obligation_type="mandatory",
                requirements=[
                    "Identify and analyse known and reasonably foreseeable risks",
                    "Estimate and evaluate risks from intended use and foreseeable misuse",
                    "Adopt risk management measures to address identified risks",
                    "Test system to identify appropriate risk management measures",
                    "Update risk management system throughout lifecycle",
                ],
                deadline="2026-08-02",
            ),
            ArticleObligation(
                article="Article 10",
                title="Data and data governance",
                summary="Training, validation, and testing data sets for high-risk AI must be subject to appropriate data governance and management practices.",
                risk_category="high",
                obligation_type="mandatory",
                requirements=[
                    "Implement data governance covering design choices, data collection processes, and preparation",
                    "Ensure training data is relevant, representative, and free of errors",
                    "Address potential biases in data sets",
                    "Use personal data only where strictly necessary for bias monitoring",
                    "Document data provenance and quality metrics",
                ],
                deadline="2026-08-02",
            ),
            ArticleObligation(
                article="Article 11",
                title="Technical documentation",
                summary="Technical documentation must be drawn up before a high-risk AI system is placed on the market and kept up to date.",
                risk_category="high",
                obligation_type="mandatory",
                requirements=[
                    "Prepare technical documentation per Annex IV before market placement",
                    "Include general description, design specifications, and development process",
                    "Document monitoring, functioning, and control of the system",
                    "Include risk management system documentation",
                    "Describe validation and testing procedures and results",
                    "Keep documentation up to date throughout system lifecycle",
                ],
                deadline="2026-08-02",
            ),
            ArticleObligation(
                article="Article 12",
                title="Record-keeping",
                summary="High-risk AI systems must be designed to enable automatic recording of events (logs) throughout their lifetime.",
                risk_category="high",
                obligation_type="mandatory",
                requirements=[
                    "Enable automatic logging of system operations",
                    "Logs must allow traceability of system functioning",
                    "Retain logs for period appropriate to intended purpose",
                    "Ensure logs are accessible to deployers and authorities",
                ],
                deadline="2026-08-02",
            ),
            ArticleObligation(
                article="Article 13",
                title="Transparency and provision of information to deployers",
                summary="High-risk AI systems must be designed to ensure sufficiently transparent operation for deployers to interpret and use outputs appropriately.",
                risk_category="high",
                obligation_type="mandatory",
                requirements=[
                    "Provide instructions for use with concise, correct, clear information",
                    "Include system capabilities and limitations",
                    "Specify intended purpose and foreseeable misuse scenarios",
                    "Describe performance metrics and known biases",
                    "Include human oversight measures",
                    "Provide expected lifetime and maintenance requirements",
                ],
                deadline="2026-08-02",
            ),
            ArticleObligation(
                article="Article 14",
                title="Human oversight",
                summary="High-risk AI systems must be designed to allow effective oversight by natural persons during use.",
                risk_category="high",
                obligation_type="mandatory",
                requirements=[
                    "Enable human understanding of system capabilities and limitations",
                    "Allow human to monitor system operation and detect anomalies",
                    "Enable human to interpret system output correctly",
                    "Allow human to override or reverse system output",
                    "Provide ability to interrupt or stop the system (stop button)",
                ],
                deadline="2026-08-02",
            ),
            ArticleObligation(
                article="Article 15",
                title="Accuracy, robustness and cybersecurity",
                summary="High-risk AI systems must achieve appropriate levels of accuracy, robustness, and cybersecurity.",
                risk_category="high",
                obligation_type="mandatory",
                requirements=[
                    "Declare and document accuracy metrics",
                    "Design for resilience to errors and inconsistencies",
                    "Implement technical redundancy solutions where appropriate",
                    "Protect against unauthorised third-party manipulation",
                    "Address vulnerabilities specific to training data (poisoning, adversarial examples)",
                ],
                deadline="2026-08-02",
            ),
            ArticleObligation(
                article="Article 16",
                title="Obligations of providers of high-risk AI systems",
                summary="Providers must ensure compliance with all Chapter III Section 2 requirements.",
                risk_category="high",
                obligation_type="mandatory",
                requirements=[
                    "Ensure system complies with Articles 8-15",
                    "Implement quality management system (Article 17)",
                    "Keep technical documentation (Article 18)",
                    "Enable automatic logging",
                    "Undergo conformity assessment before placing on market",
                    "Register in EU database",
                    "Take corrective actions when system is non-compliant",
                    "Affix CE marking",
                    "Demonstrate conformity upon request of national authority",
                ],
                deadline="2026-08-02",
            ),
            ArticleObligation(
                article="Article 17",
                title="Quality management system",
                summary="Providers must establish a quality management system ensuring compliance with the Regulation.",
                risk_category="high",
                obligation_type="mandatory",
                requirements=[
                    "Establish documented quality management policies and procedures",
                    "Include design, development and testing procedures",
                    "Include technical specifications and standards",
                    "Implement data management procedures",
                    "Include risk management process",
                    "Implement post-market monitoring system",
                    "Include procedures for incident reporting",
                    "Handle communication with competent authorities",
                    "Include resource management procedures",
                    "Establish accountability framework",
                ],
                deadline="2026-08-02",
            ),
            ArticleObligation(
                article="Article 26",
                title="Obligations of deployers",
                summary="Deployers of high-risk AI systems must use systems in accordance with instructions and take appropriate technical and organisational measures.",
                risk_category="high",
                obligation_type="mandatory",
                requirements=[
                    "Use high-risk AI in accordance with instructions of use",
                    "Ensure human oversight by trained and competent persons",
                    "Ensure input data is relevant and representative",
                    "Monitor operation of the system based on instructions",
                    "Inform provider or distributor of any serious incident",
                    "Keep logs automatically generated by the system",
                    "Carry out data protection impact assessment where required",
                    "Inform affected persons that they are subject to high-risk AI",
                ],
                deadline="2026-08-02",
            ),
            ArticleObligation(
                article="Article 27",
                title="Fundamental rights impact assessment for deployers",
                summary="Public bodies and certain private deployers must conduct a fundamental rights impact assessment before deploying high-risk AI.",
                risk_category="high",
                obligation_type="mandatory",
                requirements=[
                    "Describe deployer's processes using the AI system",
                    "Describe period and frequency of AI system use",
                    "Identify categories of natural persons affected",
                    "Assess specific risks of harm to identified groups",
                    "Describe human oversight measures",
                    "Describe measures if risks materialise",
                    "Notify market surveillance authority of assessment results",
                ],
                deadline="2026-08-02",
            ),
            ArticleObligation(
                article="Article 43",
                title="Conformity assessment",
                summary="Providers must subject high-risk AI to conformity assessment before placing on market. Route depends on whether internal control or third-party assessment is required.",
                risk_category="high",
                obligation_type="mandatory",
                requirements=[
                    "Complete conformity assessment procedure per applicable route",
                    "For Annex III systems: internal control (Annex VI) unless biometric, then third-party (Annex VII)",
                    "For Annex I systems: follow relevant sectoral conformity assessment",
                    "Repeat assessment for substantial modifications",
                    "Maintain documentation of assessment",
                ],
                deadline="2026-08-02",
            ),
            ArticleObligation(
                article="Article 47",
                title="EU declaration of conformity",
                summary="Providers must draw up a written EU declaration of conformity for each high-risk AI system.",
                risk_category="high",
                obligation_type="mandatory",
                requirements=[
                    "Include system identification and provider details",
                    "State that the declaration is issued under sole responsibility of provider",
                    "Reference applicable harmonised standards or specifications",
                    "Keep declaration up to date",
                    "Make declaration available to national authorities for 10 years",
                ],
                deadline="2026-08-02",
            ),
            ArticleObligation(
                article="Article 49",
                title="CE marking",
                summary="High-risk AI systems must bear the CE marking indicating conformity.",
                risk_category="high",
                obligation_type="mandatory",
                requirements=[
                    "Affix CE marking visibly, legibly, and indelibly",
                    "Affix before placing on the market",
                    "Include notified body identification number where applicable",
                    "Follow general principles of CE marking per Regulation (EC) No 765/2008",
                ],
                deadline="2026-08-02",
            ),
            ArticleObligation(
                article="Article 50",
                title="Transparency obligations for certain AI systems",
                summary="Specific transparency requirements for AI systems interacting with persons, generating synthetic content, or performing emotion recognition or biometric categorisation.",
                risk_category="limited",
                obligation_type="mandatory",
                requirements=[
                    "Inform natural persons they are interacting with an AI system (chatbots)",
                    "Mark AI-generated or manipulated content (deep fakes) as artificially generated",
                    "Inform persons subject to emotion recognition or biometric categorisation",
                    "Label AI-generated text published for public information purposes",
                    "Exceptions for authorised law enforcement and artistic works",
                ],
                deadline="2026-08-02",
            ),
            ArticleObligation(
                article="Article 51",
                title="Registration of high-risk AI in EU database",
                summary="Providers and deployers must register high-risk AI systems in the EU database before placing on market.",
                risk_category="high",
                obligation_type="mandatory",
                requirements=[
                    "Register system in EU database maintained by Commission",
                    "Provide required information per Annex VIII",
                    "Keep registration information up to date",
                    "Deployers who are public authorities must also register",
                ],
                deadline="2026-08-02",
            ),
            ArticleObligation(
                article="Article 53",
                title="Obligations for providers of general-purpose AI models",
                summary="All GPAI model providers must comply with transparency and documentation requirements.",
                risk_category="systemic",
                obligation_type="mandatory",
                requirements=[
                    "Draw up and keep up to date technical documentation of the model",
                    "Provide information and documentation to downstream providers",
                    "Establish policy to comply with Union copyright law",
                    "Publish sufficiently detailed summary of training data content",
                ],
                deadline="2025-08-02",
            ),
            ArticleObligation(
                article="Article 55",
                title="Obligations for providers of GPAI models with systemic risk",
                summary="GPAI models classified as having systemic risk face additional obligations beyond base GPAI requirements.",
                risk_category="systemic",
                obligation_type="mandatory",
                requirements=[
                    "Perform model evaluation including adversarial testing",
                    "Assess and mitigate possible systemic risks",
                    "Track, document, and report serious incidents to AI Office",
                    "Ensure adequate level of cybersecurity protection",
                    "Report to AI Office without undue delay upon classification",
                ],
                deadline="2025-08-02",
            ),
            ArticleObligation(
                article="Article 72",
                title="Post-market monitoring by providers",
                summary="Providers must establish and document a post-market monitoring system proportionate to the nature and risks of the AI system.",
                risk_category="high",
                obligation_type="mandatory",
                requirements=[
                    "Establish post-market monitoring system",
                    "Actively and systematically collect and analyse data from deployers",
                    "Evaluate continuous compliance with Chapter III requirements",
                    "Feed results into risk management system updates",
                    "System must be proportionate to nature of AI and risks",
                ],
                deadline="2026-08-02",
            ),
            ArticleObligation(
                article="Article 73",
                title="Reporting of serious incidents",
                summary="Providers must report serious incidents to market surveillance authorities.",
                risk_category="high",
                obligation_type="mandatory",
                requirements=[
                    "Report serious incidents to relevant market surveillance authority",
                    "Report immediately and no later than 15 days after becoming aware",
                    "Include nature of incident, identification of system, corrective actions",
                    "Serious incidents include death, serious damage to health, property, or environment",
                ],
                deadline="2026-08-02",
            ),
            ArticleObligation(
                article="Article 85",
                title="Penalties",
                summary="Defines penalty tiers for non-compliance with different provisions.",
                risk_category="high",
                obligation_type="mandatory",
                requirements=[
                    "Penalties for prohibited practices: up to 35M EUR or 7% of global turnover",
                    "Penalties for most other violations: up to 15M EUR or 3% of global turnover",
                    "Penalties for incorrect information to authorities: up to 7.5M EUR or 1.5% of turnover",
                    "SME-specific proportionate caps apply",
                    "Member States to lay down effective, proportionate, and dissuasive penalties",
                ],
                deadline="2026-08-02",
                penalties={
                    "tier_1_max_eur": 35_000_000, "tier_1_max_pct": 7.0, "tier_1_scope": "prohibited_practices",
                    "tier_2_max_eur": 15_000_000, "tier_2_max_pct": 3.0, "tier_2_scope": "most_other_obligations",
                    "tier_3_max_eur": 7_500_000, "tier_3_max_pct": 1.5, "tier_3_scope": "incorrect_information",
                },
            ),
            ArticleObligation(
                article="Article 4",
                title="AI literacy",
                summary="Providers and deployers shall ensure sufficient AI literacy among staff and persons handling AI system operation and use.",
                risk_category="minimal",
                obligation_type="mandatory",
                requirements=[
                    "Ensure staff have sufficient knowledge and understanding of AI",
                    "Take into account technical knowledge, experience, education, and context of use",
                    "Consider persons or groups affected by the AI system",
                ],
                deadline="2025-02-02",
            ),
            ArticleObligation(
                article="Article 25",
                title="Responsibilities along the AI value chain",
                summary="Third parties that modify or repurpose high-risk AI systems become subject to provider obligations.",
                risk_category="high",
                obligation_type="conditional",
                requirements=[
                    "Parties making substantial modifications assume provider obligations",
                    "Distributors, importers, deployers or other third parties must act as provider if they modify the intended purpose",
                    "Original provider must provide necessary access and cooperation",
                ],
                deadline="2026-08-02",
            ),
            ArticleObligation(
                article="Article 40",
                title="Harmonised standards",
                summary="AI systems conforming to harmonised standards benefit from presumption of conformity.",
                risk_category="high",
                obligation_type="recommended",
                requirements=[
                    "Follow harmonised standards published in the Official Journal where available",
                    "Benefit from presumption of conformity for requirements covered",
                    "Where no harmonised standards exist, follow common specifications",
                ],
                deadline="2026-08-02",
            ),
            ArticleObligation(
                article="Article 60",
                title="AI regulatory sandboxes",
                summary="Member States shall establish AI regulatory sandboxes for development, testing, and validation of innovative AI systems.",
                risk_category="minimal",
                obligation_type="mandatory",
                requirements=[
                    "Member States must establish at least one sandbox by August 2026",
                    "Sandboxes shall provide controlled environment for development and testing",
                    "Participation is voluntary for providers",
                    "Sandbox must ensure participant rights and safety",
                ],
                deadline="2026-08-02",
            ),
            ArticleObligation(
                article="Article 86",
                title="Right to explanation of individual decision-making",
                summary="Affected persons may request meaningful explanation of the role of the high-risk AI system in the decision-making procedure.",
                risk_category="high",
                obligation_type="mandatory",
                requirements=[
                    "Provide clear and meaningful information about the role of AI in the decision",
                    "Include main elements of the decision, especially the role of AI",
                    "Information shall be provided within reasonable time",
                    "Does not apply when this is already provided under EU or national law",
                ],
                deadline="2026-08-02",
            ),
            ArticleObligation(
                article="Article 95",
                title="Codes of conduct for non-high-risk AI",
                summary="Commission and Member States encourage voluntary codes of conduct for AI systems other than high-risk.",
                risk_category="minimal",
                obligation_type="optional",
                requirements=[
                    "Voluntary application of high-risk requirements to non-high-risk systems",
                    "Codes may include environmental sustainability measures",
                    "Include accessibility for persons with disabilities",
                    "Include stakeholder participation in design and development",
                    "Include diversity of development teams",
                ],
                deadline="2026-08-02",
            ),
        ]

    def _init_annexes(self) -> list:
        """Define annex requirements."""
        return [
            AnnexRequirement(
                annex="Annex I",
                title="Union harmonisation legislation — high-risk AI in regulated products",
                items=[
                    "Regulation (EU) 2017/745 — Medical devices",
                    "Regulation (EU) 2017/746 — In vitro diagnostic medical devices",
                    "Directive 2006/42/EC — Machinery",
                    "Regulation (EU) 2019/2144 — Motor vehicle type-approval",
                    "Regulation (EU) No 305/2011 — Construction products",
                    "Directive 2014/90/EU — Marine equipment",
                    "Directive (EU) 2016/797 — Railway interoperability",
                    "Regulation (EU) 2018/858 — Motor vehicle approval",
                    "Regulation (EU) 2018/1139 — Civil aviation safety",
                ],
            ),
            AnnexRequirement(
                annex="Annex III",
                title="High-risk AI systems — standalone AI applications",
                items=[
                    "1. Biometrics: remote biometric identification, biometric categorisation, emotion recognition",
                    "2. Critical infrastructure: safety components of road traffic, water/gas/heating/electricity supply",
                    "3. Education and vocational training: admission decisions, assessment, proctoring, learning disabilities",
                    "4. Employment, workers management: recruitment, job applications, performance evaluation, task allocation, termination",
                    "5. Access to essential services: creditworthiness, life/health insurance, emergency services dispatch, social benefits",
                    "6. Law enforcement: individual risk assessment, polygraph, evidence reliability, profiling for detection/investigation, crime analytics",
                    "7. Migration, asylum and border control: polygraphs, risk assessment, identity verification, asylum application processing",
                    "8. Administration of justice and democratic processes: AI assisting judicial authorities in researching and interpreting facts and law",
                ],
            ),
            AnnexRequirement(
                annex="Annex IV",
                title="Technical documentation for high-risk AI systems",
                items=[
                    "1. General description of the AI system",
                    "2. Detailed description of elements and development process",
                    "3. Detailed information about monitoring, functioning, and control",
                    "4. Description of the appropriateness of performance metrics",
                    "5. Description of the risk management system",
                    "6. Description of changes made throughout lifecycle",
                    "7. List of harmonised standards applied",
                    "8. Copy of the EU declaration of conformity",
                    "9. Description of the post-market monitoring system",
                ],
            ),
            AnnexRequirement(
                annex="Annex VII",
                title="Conformity assessment based on assessment of quality management system and technical documentation",
                items=[
                    "1. Application to notified body with system description",
                    "2. Quality management system audit by notified body",
                    "3. Technical documentation assessment",
                    "4. Surveillance activities and periodic audits",
                    "5. Amendments to quality management system require notification",
                    "6. Certificate valid for maximum 5 years (renewable)",
                ],
            ),
            AnnexRequirement(
                annex="Annex VIII",
                title="Information to be submitted for registration in EU database",
                items=[
                    "Section A — for providers: name, trade name, system name, intended purpose, status, member states of availability",
                    "Section B — for deployers: name, contact, deployer category (public/private), system identification, member state",
                    "Section C — for GPAI models: provider name, model name, release date, training compute, open-source status",
                ],
            ),
        ]

    def get_articles_by_risk(self, risk: str) -> list:
        """Get all article obligations for a given risk category."""
        return [a for a in self.articles if a.risk_category == risk]

    def get_articles_by_deadline(self, before: str) -> list:
        """Get articles with deadlines on or before the given date."""
        return [a for a in self.articles if a.deadline and a.deadline <= before]

    def get_phase_milestones(self) -> list:
        """Get all phase-in milestones in chronological order."""
        return sorted(self.milestones, key=lambda m: m.date)

    def get_annex(self, annex_id: str) -> AnnexRequirement:
        """Get a specific annex by identifier (e.g., 'Annex III')."""
        for annex in self.annexes:
            if annex.annex == annex_id:
                return annex
        return None

    def get_penalties(self) -> dict:
        """Get penalty tier information."""
        return {
            "tier_1": {
                "description": "Prohibited AI practices (Article 5)",
                "max_fine_eur": 35_000_000,
                "max_turnover_pct": 7.0,
            },
            "tier_2": {
                "description": "Most other obligations",
                "max_fine_eur": 15_000_000,
                "max_turnover_pct": 3.0,
            },
            "tier_3": {
                "description": "Supplying incorrect information to authorities",
                "max_fine_eur": 7_500_000,
                "max_turnover_pct": 1.5,
            },
            "sme_provision": "For SMEs and startups, the lower of the two amounts applies as the cap",
        }

    def get_high_risk_areas(self) -> list:
        """Get the 8 Annex III high-risk areas."""
        annex_iii = self.get_annex("Annex III")
        return annex_iii.items if annex_iii else []

    def is_high_risk(self, use_case: str) -> bool:
        """Simple keyword-based check for whether a use case is likely high-risk."""
        high_risk_keywords = [
            "biometric", "facial recognition", "emotion recognition",
            "critical infrastructure", "water supply", "electricity", "gas supply",
            "education", "admission", "grading", "proctoring",
            "recruitment", "hiring", "employment", "performance evaluation", "termination",
            "credit scoring", "insurance", "emergency services", "social benefit",
            "law enforcement", "policing", "profiling", "evidence",
            "migration", "asylum", "border control", "visa",
            "judicial", "court", "justice", "legal research",
        ]
        use_lower = use_case.lower()
        return any(kw in use_lower for kw in high_risk_keywords)

    def get_gpai_obligations(self) -> list:
        """Get obligations specific to general-purpose AI models."""
        return [a for a in self.articles if a.risk_category == "systemic"]

    def get_obligations_by_role(self, role: str) -> list:
        """Get obligations applicable to a specific role ('provider' or 'deployer')."""
        provider_articles = {"Article 9", "Article 10", "Article 11", "Article 12",
                             "Article 13", "Article 14", "Article 15", "Article 16",
                             "Article 17", "Article 43", "Article 47", "Article 49",
                             "Article 51", "Article 53", "Article 55", "Article 72",
                             "Article 73"}
        deployer_articles = {"Article 26", "Article 27", "Article 86"}
        both_articles = {"Article 4", "Article 5", "Article 50"}

        target = set()
        if role == "provider":
            target = provider_articles | both_articles
        elif role == "deployer":
            target = deployer_articles | both_articles
        else:
            target = provider_articles | deployer_articles | both_articles
        return [a for a in self.articles if a.article in target]

    def summary(self) -> str:
        """Human-readable summary of the EU AI Act encoding."""
        lines = [
            "EU Artificial Intelligence Act — Regulation (EU) 2024/1689",
            f"  Articles encoded: {len(self.articles)}",
            f"  Phase-in milestones: {len(self.milestones)}",
            f"  Annexes: {len(self.annexes)}",
            f"  Risk categories: {len(self.risk_categories)}",
            "  Penalty tiers: 3 (7%/35M, 3%/15M, 1.5%/7.5M EUR)",
            "  Key dates:",
        ]
        for m in self.get_phase_milestones():
            lines.append(f"    {m.date}: {m.description}")
        return "\n".join(lines)

    def to_json(self) -> dict:
        """Serialize the encoding to JSON-compatible dict."""
        return {
            "framework": "eu_ai_act",
            "name": "EU Artificial Intelligence Act",
            "risk_categories": self.risk_categories,
            "milestones": [
                {"date": m.date, "description": m.description,
                 "articles": m.articles, "requirements": m.requirements}
                for m in self.milestones
            ],
            "articles": [
                {"article": a.article, "title": a.title, "summary": a.summary,
                 "risk_category": a.risk_category, "obligation_type": a.obligation_type,
                 "requirements": a.requirements, "deadline": a.deadline,
                 "penalties": a.penalties}
                for a in self.articles
            ],
            "annexes": [
                {"annex": a.annex, "title": a.title, "items": a.items}
                for a in self.annexes
            ],
        }
