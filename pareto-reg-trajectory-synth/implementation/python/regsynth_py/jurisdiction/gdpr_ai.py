"""GDPR AI-relevant provisions encoding.

Encodes GDPR articles most relevant to AI systems, including automated
decision-making, data protection impact assessments, and data subject rights.
"""

from dataclasses import dataclass, field
import json


@dataclass
class GDPRProvision:
    """A GDPR provision with AI-specific relevance."""
    article: str
    title: str
    content: str
    ai_relevance: str
    requirements: list = field(default_factory=list)
    obligation_type: str = "mandatory"


class GDPRAIProvisions:
    """GDPR provisions most relevant to AI system development and deployment.

    Regulation (EU) 2016/679, in force since May 25, 2018. While not
    AI-specific, multiple provisions have significant implications for AI
    systems processing personal data.
    """

    def __init__(self):
        self.provisions = self._init_provisions()

    def _init_provisions(self) -> list:
        """Define AI-relevant GDPR provisions."""
        return [
            GDPRProvision(
                "Article 5", "Principles relating to processing of personal data",
                "Personal data shall be processed lawfully, fairly and in a transparent manner; collected for specified, explicit and legitimate purposes; adequate, relevant and limited to what is necessary; accurate and kept up to date; kept for no longer than necessary; processed with appropriate security.",
                "Core principles directly constrain AI systems. Data minimization limits training data volume. Purpose limitation restricts model reuse. Accuracy requirements apply to AI predictions affecting individuals.",
                ["Lawfulness, fairness and transparency of processing",
                 "Purpose limitation — collect for specified, explicit, legitimate purposes",
                 "Data minimization — adequate, relevant, and limited to what is necessary",
                 "Accuracy — ensure personal data is accurate and kept up to date",
                 "Storage limitation — retain no longer than necessary",
                 "Integrity and confidentiality — appropriate security measures",
                 "Accountability — controller must demonstrate compliance with all principles"]),
            GDPRProvision(
                "Article 6", "Lawfulness of processing",
                "Processing is lawful only if at least one legal basis applies: consent, contractual necessity, legal obligation, vital interests, public task, or legitimate interests.",
                "AI training on personal data requires a valid legal basis. Legitimate interest is most commonly invoked for AI but requires balancing test. Consent must be freely given, specific, informed, and unambiguous.",
                ["Identify and document lawful basis for each processing activity",
                 "Consent must be freely given, specific, informed, and unambiguous",
                 "Legitimate interest requires documented balancing test",
                 "Legal basis must be established before processing begins",
                 "Different legal bases may apply to different stages of AI lifecycle"]),
            GDPRProvision(
                "Article 9", "Processing of special categories of personal data",
                "Processing of data revealing racial origin, political opinions, religious beliefs, trade union membership, genetic data, biometric data, health data, or sex life/orientation is prohibited unless specific exceptions apply.",
                "AI systems processing biometric data, health data, or data inferring protected characteristics must satisfy Article 9 requirements. AI-inferred sensitive data may also qualify as special category data.",
                ["Explicit consent required for special category data processing",
                 "AI systems must not process special category data without valid exception",
                 "Inferred sensitive data (e.g., health predictions from non-health data) may trigger Article 9",
                 "Biometric identification systems require explicit legal basis",
                 "Health AI applications need specific authorization"]),
            GDPRProvision(
                "Article 12", "Transparent information, communication and modalities",
                "The controller shall take appropriate measures to provide information in a concise, transparent, intelligible and easily accessible form, using clear and plain language.",
                "AI systems must provide clear information about how personal data is used in model training and inference. Complex AI processing must be explained in accessible terms.",
                ["Provide information in concise, transparent, intelligible form",
                 "Use clear and plain language",
                 "Provide information in writing or electronically",
                 "Facilitate exercise of data subject rights"]),
            GDPRProvision(
                "Article 13", "Information to be provided where personal data are collected from the data subject",
                "At the time personal data is collected, the controller shall provide the data subject with information including identity, purposes, legal basis, recipients, retention period, and the existence of automated decision-making.",
                "AI developers must inform individuals when their data is collected for AI training or when they are subject to AI-based decisions.",
                ["Provide identity and contact details of controller",
                 "State purposes and legal basis for processing",
                 "Inform of recipients or categories of recipients",
                 "State retention period or criteria for determining it",
                 "Inform of existence of automated decision-making including profiling",
                 "Provide meaningful information about logic, significance, and envisaged consequences of automated processing"]),
            GDPRProvision(
                "Article 14", "Information where data not obtained from the data subject",
                "Where personal data has not been obtained from the data subject, the controller shall provide information about the source and categories of data.",
                "When AI training data is obtained from third parties or public sources, controllers must still inform data subjects about the processing.",
                ["Inform data subjects of data source",
                 "Provide within reasonable period, no later than one month",
                 "State categories of personal data concerned",
                 "Applies to AI training data from third-party sources"]),
            GDPRProvision(
                "Article 15", "Right of access by the data subject",
                "The data subject has the right to obtain confirmation of whether personal data is being processed and, where that is the case, access to the personal data and specified information.",
                "Individuals can request to know if their data was used to train AI models and what outputs AI systems produce about them. Fulfilling this for AI is technically challenging.",
                ["Confirm whether personal data is being processed",
                 "Provide access to the personal data",
                 "Provide information about purposes, categories, recipients",
                 "Inform of existence of automated decision-making",
                 "Provide copy of personal data undergoing processing",
                 "For AI: may include information about training data inclusion and model outputs"]),
            GDPRProvision(
                "Article 17", "Right to erasure ('right to be forgotten')",
                "The data subject has the right to obtain erasure of personal data without undue delay in certain circumstances, including withdrawal of consent and where data is no longer necessary.",
                "Erasure is technically challenging for AI models where personal data is embedded in model weights. May require model retraining or machine unlearning techniques. Tension with model retention.",
                ["Erase personal data without undue delay when requested and grounds apply",
                 "Grounds include: withdrawal of consent, data no longer necessary, unlawful processing",
                 "For AI: consider whether erasure requires model retraining",
                 "Machine unlearning may be needed to effectively erase from models",
                 "Document technical measures taken to achieve erasure"]),
            GDPRProvision(
                "Article 20", "Right to data portability",
                "The data subject has the right to receive their personal data in a structured, commonly used, machine-readable format and to transmit it to another controller.",
                "Applies to AI systems processing personal data based on consent or contract. AI model outputs about an individual may be portable. Challenges with AI-derived data.",
                ["Provide data in structured, commonly used, machine-readable format",
                 "Enable transmission to another controller",
                 "Applies where processing is based on consent or contract and is automated",
                 "For AI: determine which AI-derived data constitutes personal data for portability"]),
            GDPRProvision(
                "Article 21", "Right to object",
                "The data subject has the right to object to processing based on legitimate interests or public task, including profiling. The controller must cease processing unless compelling legitimate grounds are demonstrated.",
                "Individuals can object to AI profiling. Controllers must balance their AI processing interests against the individual's rights. Direct marketing profiling must cease immediately upon objection.",
                ["Respect right to object to processing based on legitimate interest or public task",
                 "Cease processing unless compelling legitimate grounds demonstrated",
                 "For direct marketing profiling: cease immediately upon objection",
                 "Inform data subjects of right to object clearly",
                 "For AI: objection to profiling may require excluding individual from AI processing"]),
            GDPRProvision(
                "Article 22", "Automated individual decision-making, including profiling",
                "The data subject has the right not to be subject to a decision based solely on automated processing, including profiling, which produces legal effects or similarly significantly affects them. Exceptions for contractual necessity, legal authorization, or explicit consent.",
                "Central provision for AI. Prohibits fully automated decisions with significant effects unless exceptions apply. Requires meaningful human involvement for high-impact AI decisions. Right to obtain human intervention, express views, and contest the decision.",
                ["Do not subject individuals to solely automated decisions with legal/significant effects",
                 "Exceptions: contractual necessity, Union/Member State law, explicit consent",
                 "Where exception applies, implement suitable safeguards",
                 "Safeguards include: right to obtain human intervention",
                 "Right to express point of view",
                 "Right to contest the automated decision",
                 "Provide meaningful information about logic involved",
                 "Do not base automated decisions on special category data unless explicit consent or substantial public interest"]),
            GDPRProvision(
                "Article 25", "Data protection by design and by default",
                "The controller shall implement appropriate technical and organisational measures designed to implement data-protection principles effectively, both at the time of determination of processing means and at time of processing itself.",
                "AI systems must incorporate data protection from the design phase. Privacy-enhancing technologies (PETs), differential privacy, federated learning, and anonymisation should be considered during AI system design.",
                ["Implement data protection principles by design in AI systems",
                 "Apply data minimization by default",
                 "Consider privacy-enhancing technologies in AI architecture",
                 "Limit personal data processed to what is necessary by default",
                 "Implement state-of-the-art technical measures"]),
            GDPRProvision(
                "Article 30", "Records of processing activities",
                "Each controller shall maintain a record of processing activities under its responsibility, containing specified information including purposes, data categories, recipients, and retention periods.",
                "AI processing activities must be documented including model training, inference, and any profiling. Records must cover the full AI data pipeline.",
                ["Maintain record of all processing activities",
                 "Include: purposes, categories of data subjects, data categories",
                 "Include: recipients, transfers to third countries, retention periods",
                 "Include: description of technical and organisational security measures",
                 "For AI: document model training data sources and processing pipeline"]),
            GDPRProvision(
                "Article 32", "Security of processing",
                "The controller and processor shall implement appropriate technical and organisational measures to ensure a level of security appropriate to the risk.",
                "AI systems must be secured against unauthorized access, data breaches, and adversarial attacks. Model security includes protection of training data, model weights, and inference pipelines.",
                ["Implement security appropriate to the risk",
                 "Consider: pseudonymisation and encryption of personal data",
                 "Ensure ongoing confidentiality, integrity, availability and resilience",
                 "Ability to restore data access in the event of incident",
                 "Process for regularly testing and evaluating security measures",
                 "For AI: protect against adversarial attacks and model extraction"]),
            GDPRProvision(
                "Article 35", "Data protection impact assessment",
                "Where a type of processing is likely to result in a high risk to the rights and freedoms of natural persons, the controller shall carry out an assessment of the impact of the envisaged processing operations on the protection of personal data.",
                "DPIAs are required for many AI use cases, especially systematic profiling with significant effects, large-scale processing of special categories, and systematic monitoring of publicly accessible areas.",
                ["Conduct DPIA before processing that is likely to result in high risk",
                 "Required for: systematic profiling with significant effects",
                 "Required for: large-scale processing of special categories",
                 "Required for: systematic monitoring of publicly accessible areas",
                 "Assessment must include: description of processing, necessity, proportionality",
                 "Assessment must include: risk assessment and mitigation measures",
                 "Consult DPO where designated",
                 "Consult supervisory authority if high risk cannot be mitigated (Art 36)"]),
            GDPRProvision(
                "Article 36", "Prior consultation",
                "The controller shall consult the supervisory authority prior to processing where a data protection impact assessment indicates that the processing would result in a high risk in the absence of measures taken by the controller to mitigate the risk.",
                "If a DPIA for an AI system shows unmitigable high risk, the supervisory authority must be consulted before deployment.",
                ["Consult supervisory authority if DPIA shows residual high risk",
                 "Provide authority with DPIA, DPO details, and processing information",
                 "Authority may provide written advice within 8 weeks (extendable)"]),
            GDPRProvision(
                "Article 37", "Designation of the data protection officer",
                "The controller and the processor shall designate a data protection officer where processing is carried out by a public authority, core activities require regular and systematic monitoring, or core activities involve large-scale processing of special categories.",
                "Organizations deploying AI at scale, especially for monitoring or profiling, likely require a DPO. DPO should have expertise in AI-related data protection issues.",
                ["Designate DPO where: public authority, regular monitoring at scale, or large-scale special category data",
                 "Ensure DPO is involved in all AI data protection matters",
                 "DPO should have AI data protection expertise"]),
            GDPRProvision(
                "Article 44", "General principle for transfers",
                "Any transfer of personal data to a third country or an international organisation shall take place only if the conditions laid down in Chapter V are complied with by the controller and processor.",
                "AI services that transfer personal data across borders — including cloud-based training or inference — must comply with Chapter V transfer mechanisms.",
                ["Ensure a valid transfer mechanism exists before cross-border data transfer",
                 "Apply transfer rules to all AI processing across borders",
                 "Covers both training data transfers and inference data flows"]),
            GDPRProvision(
                "Article 45", "Transfers on the basis of an adequacy decision",
                "A transfer of personal data to a third country may take place where the European Commission has decided that the third country ensures an adequate level of protection.",
                "AI providers can transfer data freely to countries with EU adequacy decisions (Japan, UK, South Korea, EU-US Data Privacy Framework).",
                ["Verify destination country has current adequacy decision",
                 "Monitor for changes to adequacy status",
                 "No specific authorisation required for transfer to adequate countries"],
                "conditional"),
            GDPRProvision(
                "Article 46", "Transfers subject to appropriate safeguards",
                "In the absence of an adequacy decision, a controller or processor may transfer personal data only if appropriate safeguards are provided, including standard contractual clauses (SCCs), binding corporate rules (BCRs), or approved codes of conduct.",
                "Most AI providers transferring EU personal data to non-adequate countries must use SCCs, BCRs, or similar mechanisms, plus conduct a Transfer Impact Assessment per Schrems II.",
                ["Use standard contractual clauses (SCCs) adopted by the Commission",
                 "Or binding corporate rules (BCRs) for intra-group transfers",
                 "Or approved codes of conduct with binding commitments",
                 "Conduct Transfer Impact Assessment (Schrems II requirement)",
                 "Implement supplementary measures where needed"],
                "conditional"),
            GDPRProvision(
                "Article 47", "Binding corporate rules",
                "Binding corporate rules shall be approved by the competent supervisory authority and shall specify the application of general data protection principles, rights of data subjects, and mechanisms for ensuring compliance.",
                "AI multinationals may use BCRs for intra-group transfers of training data and model outputs across jurisdictions.",
                ["BCRs must be legally binding and enforced by every member of the group",
                 "Must specify data protection safeguards and data subject rights",
                 "Must include complaint and dispute resolution procedures",
                 "BCRs subject to supervisory authority approval"],
                "conditional"),
            GDPRProvision(
                "Article 48", "Transfers not authorised by Union law",
                "Any judgment of a court or tribunal and any decision of an administrative authority of a third country requiring transfer of personal data may only be recognised or enforceable if based on an international agreement.",
                "Prevents AI providers from complying with foreign government data access orders without a valid international agreement. Relevant to cloud-hosted AI services.",
                ["Third-country court orders alone do not authorise transfer",
                 "International agreement required for recognition of foreign orders",
                 "AI providers must assess foreign government access risks"]),
            GDPRProvision(
                "Article 49", "Derogations for specific situations",
                "In the absence of an adequacy decision or appropriate safeguards, a transfer may take place only if the data subject has explicitly consented, the transfer is necessary for contractual performance, or for important reasons of public interest, among other derogations.",
                "Limited fallback for AI data transfers. Derogations are narrow and cannot be used for systematic or large-scale transfers. Not suitable for routine AI training data flows.",
                ["Explicit consent after being informed of risks",
                 "Transfer necessary for performance of contract with data subject",
                 "Transfer necessary for important reasons of public interest",
                 "Derogations must be interpreted restrictively",
                 "Not suitable for repetitive, large-scale, or systematic transfers"],
                "conditional"),
        ]

    def get_obligations(self) -> list:
        """Generate obligation records from all provisions."""
        obligations = []
        for prov in self.provisions:
            obligations.append({
                "id": f"GDPR-{prov.article.replace(' ', '-')}",
                "framework_id": "gdpr",
                "article": prov.article,
                "title": prov.title,
                "description": prov.content,
                "obligation_type": prov.obligation_type,
                "ai_relevance": prov.ai_relevance,
                "requirements": prov.requirements,
            })
        return obligations

    def get_ai_relevant_articles(self) -> list:
        """Get all articles with AI-specific relevance (alias for get_ai_relevant_provisions)."""
        return self.get_ai_relevant_provisions()

    def get_provision(self, article: str) -> GDPRProvision:
        """Get a provision by article reference (e.g., 'Article 22')."""
        for prov in self.provisions:
            if prov.article == article:
                return prov
        return None

    def get_ai_relevant_provisions(self) -> list:
        """Get all provisions with AI-specific relevance."""
        return list(self.provisions)

    def get_automated_decision_provisions(self) -> list:
        """Get provisions related to automated decision-making."""
        auto_articles = {"Article 22", "Article 13", "Article 14", "Article 15", "Article 21"}
        return [p for p in self.provisions if p.article in auto_articles]

    def get_dpia_requirements(self) -> list:
        """Get DPIA requirements."""
        return [
            "Conduct DPIA before high-risk processing (Article 35)",
            "Mandatory for: systematic and extensive profiling with significant effects",
            "Mandatory for: large-scale processing of special categories (Art 9/10)",
            "Mandatory for: systematic monitoring of publicly accessible areas",
            "Include: systematic description of processing operations and purposes",
            "Include: assessment of necessity and proportionality",
            "Include: assessment of risks to rights and freedoms of data subjects",
            "Include: measures to address risks including safeguards, security, and mechanisms",
            "Consult DPO during DPIA process",
            "Consult supervisory authority if residual high risk (Article 36)",
            "Review and update DPIA when processing operations change",
        ]

    def get_transparency_requirements(self) -> list:
        """Get AI transparency requirements under GDPR."""
        return [
            "Provide clear, plain language information about AI processing (Art 12)",
            "Inform of existence of automated decision-making including profiling (Art 13/14)",
            "Provide meaningful information about the logic involved (Art 13(2)(f), 14(2)(g))",
            "Explain significance and envisaged consequences of automated processing (Art 13/14)",
            "Provide information about data sources when not obtained from subject (Art 14)",
            "Enable access to personal data and processing information (Art 15)",
            "For automated decisions: provide right to human intervention (Art 22)",
        ]

    def get_data_subject_rights(self) -> list:
        """Get provisions establishing data subject rights relevant to AI."""
        rights_articles = {"Article 15", "Article 17", "Article 20", "Article 21", "Article 22"}
        return [p for p in self.provisions if p.article in rights_articles]

    def get_controller_obligations(self) -> list:
        """Get provisions establishing controller obligations for AI."""
        controller_articles = {"Article 5", "Article 6", "Article 25", "Article 30",
                               "Article 32", "Article 35", "Article 37"}
        return [p for p in self.provisions if p.article in controller_articles]

    def get_conflicts_with_ai(self) -> list:
        """Get known tensions between GDPR requirements and AI practices."""
        return [
            {
                "conflict": "Data minimization vs. training data needs",
                "gdpr_article": "Article 5(1)(c)",
                "description": "GDPR requires data to be limited to what is necessary. AI models, especially large language models, often benefit from vast training datasets.",
                "impact": "May restrict quantity and scope of training data",
                "mitigations": ["Demonstrate necessity of data volume for model quality",
                                "Use synthetic data or data augmentation",
                                "Apply differential privacy techniques",
                                "Document proportionality assessment"],
            },
            {
                "conflict": "Right to erasure vs. model weight persistence",
                "gdpr_article": "Article 17",
                "description": "Individuals can request erasure, but personal data absorbed into model weights cannot easily be removed without full retraining.",
                "impact": "May require model retraining or machine unlearning",
                "mitigations": ["Implement machine unlearning techniques",
                                "Design systems for efficient retraining",
                                "Use federated learning to reduce centralized data retention",
                                "Document why erasure from model is technically challenging"],
            },
            {
                "conflict": "Purpose limitation vs. model transfer/reuse",
                "gdpr_article": "Article 5(1)(b)",
                "description": "Data collected for one purpose should not be reused for incompatible purposes. Transfer learning and foundation model fine-tuning may violate this.",
                "impact": "Restricts reuse of personal data across different AI applications",
                "mitigations": ["Conduct compatibility assessment for new purposes",
                                "Anonymize data before transfer",
                                "Obtain fresh consent for new purposes",
                                "Use synthetic data for transfer learning"],
            },
            {
                "conflict": "Explainability vs. model complexity",
                "gdpr_article": "Article 22/Article 13(2)(f)",
                "description": "GDPR requires meaningful information about the logic of automated decisions. Complex deep learning models are difficult to explain.",
                "impact": "May limit use of opaque models for decisions affecting individuals",
                "mitigations": ["Use inherently interpretable models where feasible",
                                "Apply post-hoc explanation techniques (SHAP, LIME)",
                                "Provide process-level rather than model-level explanations",
                                "Ensure human review for high-impact decisions"],
            },
            {
                "conflict": "Accuracy principle vs. probabilistic AI outputs",
                "gdpr_article": "Article 5(1)(d)",
                "description": "GDPR requires personal data to be accurate. AI predictions are probabilistic and may produce inaccurate results about individuals.",
                "impact": "AI-derived personal data must be verified for accuracy",
                "mitigations": ["Validate model accuracy metrics",
                                "Provide mechanisms for individuals to correct inaccurate outputs",
                                "Do not treat AI predictions as ground truth",
                                "Document confidence levels and error rates"],
            },
            {
                "conflict": "Storage limitation vs. model lifecycle",
                "gdpr_article": "Article 5(1)(e)",
                "description": "Personal data should not be kept longer than necessary. AI model lifecycle may extend well beyond original data collection period.",
                "impact": "May require data deletion schedules that conflict with model maintenance",
                "mitigations": ["Define clear retention periods for training data",
                                "Separate model deployment from training data storage",
                                "Use anonymization once training is complete",
                                "Document justification for any extended retention"],
            },
        ]

    def map_to_eu_ai_act(self) -> dict:
        """Map GDPR provisions to EU AI Act articles."""
        return {
            "Article 5 (principles)": {"eu_articles": ["Article 10"], "concept": "Data governance principles"},
            "Article 6 (lawful basis)": {"eu_articles": [], "concept": "No direct equivalent (complementary)"},
            "Article 9 (special categories)": {"eu_articles": ["Article 10(5)"], "concept": "Special data processing in bias detection"},
            "Article 13-14 (transparency)": {"eu_articles": ["Article 13", "Article 50"], "concept": "Transparency obligations"},
            "Article 22 (automated decisions)": {"eu_articles": ["Article 14", "Article 86"], "concept": "Human oversight and right to explanation"},
            "Article 25 (by design)": {"eu_articles": ["Article 9", "Article 15"], "concept": "Risk management by design"},
            "Article 35 (DPIA)": {"eu_articles": ["Article 27", "Article 9"], "concept": "Impact assessment"},
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "GDPR AI-Relevant Provisions",
            f"  Provisions encoded: {len(self.provisions)}",
            "  Framework type: Binding",
            "  In force: May 25, 2018",
            "  Maximum penalty: 20M EUR or 4% global turnover",
            "  Key AI provisions:",
            "    - Automated decision-making (Article 22)",
            "    - Data protection impact assessment (Article 35)",
            "    - Transparency (Articles 12-14)",
            "    - Data subject rights (Articles 15-22)",
            "    - Data protection by design (Article 25)",
            f"  Known AI tensions: {len(self.get_conflicts_with_ai())}",
        ]
        return "\n".join(lines)

    def to_json(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "framework": "gdpr_ai",
            "name": "GDPR AI-Relevant Provisions",
            "in_force": "2018-05-25",
            "type": "binding",
            "region": "EU",
            "provisions": [
                {"article": p.article, "title": p.title, "content": p.content,
                 "ai_relevance": p.ai_relevance, "requirements": p.requirements,
                 "obligation_type": p.obligation_type}
                for p in self.provisions
            ],
            "ai_conflicts": self.get_conflicts_with_ai(),
        }


# Alias matching the conventional class name
GDPR_AI = GDPRAIProvisions
