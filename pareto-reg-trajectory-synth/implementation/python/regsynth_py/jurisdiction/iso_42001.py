"""ISO/IEC 42001:2023 AI Management System encoding.

Provides clauses and Annex A controls for AI management system certification.
"""

from dataclasses import dataclass, field
import json


@dataclass
class Clause:
    """An ISO 42001 clause."""
    id: str
    title: str
    description: str
    requirements: list = field(default_factory=list)
    mandatory: bool = True


@dataclass
class Control:
    """An Annex A control."""
    id: str
    title: str
    description: str
    objective: str
    guidance: list = field(default_factory=list)


class ISO42001:
    """ISO/IEC 42001:2023 — Artificial Intelligence Management System (AIMS).

    Published December 18, 2023. Specifies requirements for establishing,
    implementing, maintaining, and continually improving an AI management
    system within the context of an organization.
    """

    def __init__(self):
        self.clauses = self._init_clauses()
        self.controls = self._init_controls()

    def _init_clauses(self) -> list:
        """Define ISO 42001 clauses with real subclauses and requirements."""
        return [
            # Clause 4: Context of the organization
            Clause("4.1", "Understanding the organization and its context",
                   "Determine external and internal issues relevant to the organization's purpose that affect its ability to achieve the intended outcomes of its AIMS.",
                   ["Identify external issues: legal, regulatory, technological, competitive, market, cultural, social, economic",
                    "Identify internal issues: values, culture, knowledge, performance, organizational structure",
                    "Determine impact of AI-specific factors including ethical considerations and societal impact",
                    "Consider the state of the art in AI technologies and practices"]),
            Clause("4.2", "Understanding the needs and expectations of interested parties",
                   "Determine interested parties relevant to the AIMS, their requirements, and which will be addressed through the AIMS.",
                   ["Identify interested parties: customers, regulators, employees, affected individuals, society",
                    "Determine relevant requirements of interested parties",
                    "Identify legal, regulatory, and contractual requirements related to AI",
                    "Monitor and review information about interested parties and their requirements"]),
            Clause("4.3", "Determining the scope of the AI management system",
                   "Determine boundaries and applicability of the AIMS to establish its scope.",
                   ["Consider external and internal issues from 4.1",
                    "Consider requirements from 4.2",
                    "Define the scope in terms of AI systems, processes, and organizational units",
                    "Document and make the scope available"]),
            Clause("4.4", "AI management system",
                   "Establish, implement, maintain, and continually improve an AIMS including needed processes and interactions.",
                   ["Define processes needed for the AIMS and their interactions",
                    "Determine process inputs and expected outputs",
                    "Determine resources needed for processes",
                    "Implement processes according to this document"]),

            # Clause 5: Leadership
            Clause("5.1", "Leadership and commitment",
                   "Top management shall demonstrate leadership and commitment to the AIMS.",
                   ["Ensure AI policy and objectives are established and compatible with strategic direction",
                    "Ensure integration of AIMS requirements into business processes",
                    "Ensure resources needed are available",
                    "Communicate importance of effective AI management and conforming to AIMS requirements",
                    "Ensure AIMS achieves intended outcomes",
                    "Direct and support persons to contribute to AIMS effectiveness",
                    "Promote continual improvement",
                    "Support relevant management roles to demonstrate leadership in their areas"]),
            Clause("5.2", "AI policy",
                   "Top management shall establish an AI policy appropriate to the organization's purpose.",
                   ["Include commitment to satisfying applicable requirements",
                    "Include commitment to continual improvement of the AIMS",
                    "Provide framework for setting AI objectives",
                    "Include commitment to responsible AI development and use",
                    "Communicate policy within the organization",
                    "Make policy available to relevant interested parties"]),
            Clause("5.3", "Organizational roles, responsibilities and authorities",
                   "Top management shall ensure responsibilities and authorities for AIMS roles are assigned and communicated.",
                   ["Assign responsibility for ensuring AIMS conformity to this document",
                    "Assign responsibility for reporting on AIMS performance to top management",
                    "Ensure AIMS roles include AI-specific competencies",
                    "Document and communicate roles and responsibilities"]),

            # Clause 6: Planning
            Clause("6.1", "Actions to address risks and opportunities",
                   "Determine risks and opportunities that need to be addressed to ensure the AIMS achieves its intended outcomes.",
                   ["Consider issues from 4.1 and requirements from 4.2",
                    "Determine risks and opportunities including AI system impact assessment",
                    "Plan actions to address risks and opportunities",
                    "Integrate and implement actions into AIMS processes",
                    "Evaluate effectiveness of actions",
                    "Include AI-specific risks: bias, privacy, safety, security, transparency"]),
            Clause("6.2", "AI objectives and planning to achieve them",
                   "Establish AI objectives at relevant functions, levels, and processes.",
                   ["Objectives shall be consistent with AI policy",
                    "Objectives shall be measurable where practicable",
                    "Consider applicable requirements",
                    "Determine what will be done, resources needed, responsibilities, timelines, and evaluation methods",
                    "Include responsible AI objectives: fairness, transparency, accountability"]),
            Clause("6.3", "Planning of changes",
                   "When changes to the AIMS are needed, they shall be carried out in a planned manner.",
                   ["Consider the purpose of the changes and potential consequences",
                    "Consider integrity of the AIMS",
                    "Consider availability of resources",
                    "Consider allocation or reallocation of responsibilities and authorities"]),

            # Clause 7: Support
            Clause("7.1", "Resources",
                   "Determine and provide resources needed for the AIMS.",
                   ["Identify resource requirements for AIMS establishment and maintenance",
                    "Include computing resources for AI system lifecycle",
                    "Include data resources and data management infrastructure",
                    "Provide adequate human resources with AI expertise"]),
            Clause("7.2", "Competence",
                   "Determine necessary competence of persons doing work under AIMS control that affects AI system performance.",
                   ["Determine competence requirements for AI development and operations",
                    "Ensure persons are competent on the basis of education, training, or experience",
                    "Take actions to acquire necessary competence including AI-specific skills",
                    "Retain documented evidence of competence"]),
            Clause("7.3", "Awareness",
                   "Persons doing work under AIMS control shall be aware of the AI policy, their contribution, and implications of non-conformance.",
                   ["Communicate AI policy to all relevant persons",
                    "Ensure awareness of responsible AI principles",
                    "Communicate implications of not conforming with AIMS requirements",
                    "Include awareness of AI-specific risks and ethical considerations"]),
            Clause("7.4", "Communication",
                   "Determine internal and external communications relevant to the AIMS.",
                   ["Determine what to communicate about AI systems and their impacts",
                    "Determine when, with whom, and how to communicate",
                    "Include communication about AI incidents and performance",
                    "Establish channels for stakeholder feedback on AI systems"]),
            Clause("7.5", "Documented information",
                   "The AIMS shall include documented information required by this document and needed for AIMS effectiveness.",
                   ["Create and update documented information as needed",
                    "Control documented information: distribution, access, retrieval, use, storage",
                    "Include AI-specific documentation: model cards, data sheets, impact assessments",
                    "Retain documented information as evidence of AI system lifecycle activities"]),

            # Clause 8: Operation
            Clause("8.1", "Operational planning and control",
                   "Plan, implement, and control processes needed to meet AIMS requirements.",
                   ["Establish criteria for AI processes",
                    "Implement control of processes in accordance with criteria",
                    "Control planned changes and review consequences of unintended changes",
                    "Ensure outsourced processes are controlled"]),
            Clause("8.2", "AI risk assessment",
                   "Perform AI risk assessment at planned intervals or when significant changes are proposed.",
                   ["Apply AI risk assessment criteria established in planning",
                    "Ensure assessments produce consistent, valid, and comparable results",
                    "Identify risks associated with AI systems including societal impacts",
                    "Analyse and evaluate identified AI risks",
                    "Retain documented information of assessment results"]),
            Clause("8.3", "AI risk treatment",
                   "Implement the AI risk treatment plan.",
                   ["Select appropriate AI risk treatment options",
                    "Determine all controls necessary to implement treatment options",
                    "Compare controls with Annex A to verify none are omitted",
                    "Produce a Statement of Applicability",
                    "Formulate an AI risk treatment plan",
                    "Obtain risk owners' approval of the treatment plan"]),
            Clause("8.4", "AI system impact assessment",
                   "Conduct impact assessments for AI systems considering affected individuals and groups.",
                   ["Assess potential impacts on individuals, groups, and society",
                    "Consider impacts on fundamental rights and freedoms",
                    "Evaluate potential for discrimination and bias",
                    "Assess privacy and data protection impacts",
                    "Document assessment results and mitigations",
                    "Review assessments when changes to AI systems occur"]),

            # Clause 9: Performance evaluation
            Clause("9.1", "Monitoring, measurement, analysis and evaluation",
                   "Determine what needs to be monitored and measured for the AIMS.",
                   ["Determine monitoring and measurement methods including AI performance metrics",
                    "Determine when monitoring and measurement shall be performed",
                    "Determine when results shall be analysed and evaluated",
                    "Retain documented evidence of monitoring and measurement results",
                    "Evaluate AIMS performance and effectiveness",
                    "Include AI-specific metrics: accuracy, fairness, robustness, explainability"]),
            Clause("9.2", "Internal audit",
                   "Conduct internal audits at planned intervals.",
                   ["Audit conformity to organization's own AIMS requirements and this document",
                    "Audit effective implementation and maintenance",
                    "Plan audit programme considering AI system importance and risks",
                    "Define audit criteria and scope for each audit",
                    "Select auditors with AI management competence",
                    "Report audit results to relevant management"]),
            Clause("9.3", "Management review",
                   "Top management shall review the AIMS at planned intervals.",
                   ["Consider status of actions from previous reviews",
                    "Consider changes in external and internal issues",
                    "Consider AIMS performance including nonconformities, monitoring results, audit results",
                    "Consider opportunities for continual improvement",
                    "Include review of AI system performance and impact assessment results",
                    "Document decisions and actions from management review"]),

            # Clause 10: Improvement
            Clause("10.1", "Nonconformity and corrective action",
                   "React to nonconformities by taking action, dealing with consequences, and eliminating causes.",
                   ["Take action to control and correct nonconformity",
                    "Evaluate need for action to eliminate causes of nonconformity",
                    "Implement any action needed",
                    "Review effectiveness of corrective action taken",
                    "Make changes to AIMS if necessary",
                    "For AI-related nonconformities, assess impact on affected parties"]),
            Clause("10.2", "Continual improvement",
                   "Continually improve the suitability, adequacy, and effectiveness of the AIMS.",
                   ["Identify opportunities for improvement through monitoring and review",
                    "Implement improvements to AI management processes",
                    "Consider technological advances in AI and evolving best practices",
                    "Integrate lessons learned from AI system incidents and near-misses",
                    "Benchmark against industry standards and peer organizations"]),
        ]

    def _init_controls(self) -> list:
        """Define Annex A controls."""
        return [
            Control("A.2", "AI Policies",
                    "Policies for artificial intelligence providing management direction and support.",
                    "Provide management direction and support for AI in accordance with business requirements and relevant laws and regulations.",
                    ["Develop and publish AI-specific policies",
                     "Review policies at planned intervals or when significant changes occur",
                     "Align AI policies with organizational strategy and values",
                     "Include principles for responsible AI development and use",
                     "Address ethical considerations in AI policy"]),
            Control("A.3", "Internal Organization",
                    "Establish a management framework to initiate and control the implementation of AI within the organization.",
                    "Ensure clear governance structure for AI management.",
                    ["Define roles for AI oversight and governance",
                     "Segregate duties where appropriate to reduce risk of unauthorized AI use",
                     "Maintain contact with relevant authorities and professional bodies",
                     "Address AI management in project management processes",
                     "Establish AI ethics committee or review board"]),
            Control("A.4", "Resources for AI Systems",
                    "Ensure appropriate resources are identified and managed for AI systems.",
                    "Identify organizational resources related to AI systems and ensure appropriate management.",
                    ["Inventory AI assets including models, data, and infrastructure",
                     "Classify AI resources based on criticality and risk",
                     "Ensure appropriate computational and storage resources",
                     "Manage AI-specific talent and expertise",
                     "Plan for resource scaling as AI systems grow"]),
            Control("A.5", "AI System Impact Assessment",
                    "Conduct assessments of AI system impacts on individuals and society.",
                    "Systematically assess potential positive and negative impacts of AI systems.",
                    ["Establish impact assessment methodology",
                     "Assess impacts before deployment and periodically during operation",
                     "Consider impacts on human rights, safety, and wellbeing",
                     "Engage affected stakeholders in assessment process",
                     "Document and communicate assessment results",
                     "Use assessment results to inform risk treatment decisions"]),
            Control("A.6", "AI System Lifecycle",
                    "Manage AI systems throughout their lifecycle from conception to retirement.",
                    "Ensure AI systems are managed throughout all lifecycle stages.",
                    ["Define lifecycle stages: design, development, deployment, operation, retirement",
                     "Establish quality gates between lifecycle stages",
                     "Implement version control and change management for AI models",
                     "Ensure traceability from requirements through deployment",
                     "Plan and execute system retirement including data handling",
                     "Document decisions at each lifecycle stage"]),
            Control("A.7", "Data for AI Systems",
                    "Manage data used in and by AI systems.",
                    "Ensure data quality, appropriateness, and integrity for AI systems.",
                    ["Establish data governance processes for AI data",
                     "Assess data quality and fitness for purpose",
                     "Manage data provenance and lineage",
                     "Address data bias and representativeness",
                     "Ensure data privacy and protection requirements are met",
                     "Manage data retention and disposal",
                     "Document data sources, processing, and transformations"]),
            Control("A.8", "AI System Monitoring and Measurement",
                    "Monitor AI system performance and measure against objectives.",
                    "Ensure ongoing appropriate performance and behavior of AI systems.",
                    ["Define performance metrics and thresholds for AI systems",
                     "Implement continuous monitoring mechanisms",
                     "Monitor for model drift and performance degradation",
                     "Track fairness and bias metrics over time",
                     "Establish alerting for anomalous AI system behavior",
                     "Document and analyze monitoring results",
                     "Trigger reassessment when thresholds are breached"]),
            Control("A.9", "Third-party and Supplier Relations",
                    "Ensure appropriate management of AI-related third-party relationships.",
                    "Manage risks from third-party AI components, services, and data.",
                    ["Assess AI risks in supplier and partner relationships",
                     "Include AI-specific requirements in contracts and agreements",
                     "Monitor third-party AI service performance and compliance",
                     "Ensure third-party AI components meet organizational AI policies",
                     "Plan for third-party service transitions and continuity",
                     "Verify third-party data sources meet quality and ethical standards"]),
        ]

    def get_clause(self, clause_id: str) -> Clause:
        """Get a clause by ID (e.g., '4.1', '8.2')."""
        for clause in self.clauses:
            if clause.id == clause_id:
                return clause
        return None

    def get_obligations(self) -> list:
        """Generate obligation records from all clauses.

        Each mandatory clause produces a mandatory obligation;
        non-mandatory clauses produce recommended obligations.
        """
        obligations = []
        for clause in self.clauses:
            obligations.append({
                "id": f"ISO42001-{clause.id}",
                "framework_id": "iso_42001",
                "clause": clause.id,
                "title": clause.title,
                "description": clause.description,
                "obligation_type": "mandatory" if clause.mandatory else "recommended",
                "requirements": clause.requirements,
            })
        return obligations

    def get_control(self, control_id: str) -> Control:
        """Get an Annex A control by ID (e.g., 'A.5')."""
        for control in self.controls:
            if control.id == control_id:
                return control
        return None

    def get_mandatory_clauses(self) -> list:
        """Get all mandatory clauses (all normative clauses 4-10 are mandatory for certification)."""
        return [c for c in self.clauses if c.mandatory]

    def get_all_controls(self) -> list:
        """Get all Annex A controls."""
        return list(self.controls)

    def get_certification_requirements(self) -> list:
        """Get the key requirements for ISO 42001 certification."""
        return [
            "Establish, implement, maintain, and continually improve an AI management system",
            "Define scope covering all AI systems within organizational boundaries",
            "Obtain top management commitment and establish AI policy",
            "Conduct AI risk assessment and implement risk treatment plan",
            "Conduct AI system impact assessments for all significant AI systems",
            "Produce Statement of Applicability for Annex A controls",
            "Implement Annex A controls selected through risk treatment",
            "Establish monitoring, measurement, and evaluation processes",
            "Conduct internal audits and management reviews at planned intervals",
            "Demonstrate continual improvement of the AIMS",
            "Maintain documented information as evidence of conformity",
            "Pass certification audit by accredited certification body",
        ]

    def map_to_eu_ai_act(self) -> dict:
        """Map ISO 42001 clauses to EU AI Act articles."""
        return {
            "4.1-4.4": {"eu_articles": ["Article 9"], "concept": "Risk management context"},
            "5.1-5.3": {"eu_articles": ["Article 16", "Article 17"], "concept": "Provider obligations and QMS"},
            "6.1": {"eu_articles": ["Article 9"], "concept": "Risk management planning"},
            "7.2": {"eu_articles": ["Article 4"], "concept": "AI literacy and competence"},
            "7.5": {"eu_articles": ["Article 11", "Article 12"], "concept": "Technical documentation and records"},
            "8.2": {"eu_articles": ["Article 9"], "concept": "Risk assessment"},
            "8.4": {"eu_articles": ["Article 27"], "concept": "Fundamental rights impact assessment"},
            "9.1": {"eu_articles": ["Article 72"], "concept": "Post-market monitoring"},
            "9.2": {"eu_articles": ["Article 43"], "concept": "Conformity assessment"},
            "10.1": {"eu_articles": ["Article 16", "Article 73"], "concept": "Corrective action and incident reporting"},
            "A.5": {"eu_articles": ["Article 27"], "concept": "Impact assessment"},
            "A.6": {"eu_articles": ["Article 9", "Article 17"], "concept": "Lifecycle management"},
            "A.7": {"eu_articles": ["Article 10"], "concept": "Data governance"},
            "A.8": {"eu_articles": ["Article 72"], "concept": "Monitoring"},
            "A.9": {"eu_articles": ["Article 25"], "concept": "Value chain responsibilities"},
        }

    def summary(self) -> str:
        """Human-readable summary."""
        mandatory = len(self.get_mandatory_clauses())
        lines = [
            "ISO/IEC 42001:2023 — AI Management System",
            f"  Clauses: {len(self.clauses)} ({mandatory} mandatory)",
            f"  Annex A Controls: {len(self.controls)}",
            "  Framework type: Voluntary (certification-based)",
            "  Published: December 18, 2023",
            "  Certification: By accredited bodies",
            "  Clause structure:",
        ]
        current_section = ""
        for clause in self.clauses:
            section = clause.id.split(".")[0]
            section_names = {"4": "Context", "5": "Leadership", "6": "Planning",
                             "7": "Support", "8": "Operation", "9": "Performance evaluation",
                             "10": "Improvement"}
            if section != current_section:
                current_section = section
                lines.append(f"    Clause {section}: {section_names.get(section, '')}")
            lines.append(f"      {clause.id}: {clause.title}")
        return "\n".join(lines)

    def to_json(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "framework": "iso_42001",
            "name": "ISO/IEC 42001:2023",
            "published": "2023-12-18",
            "type": "voluntary",
            "clauses": [
                {"id": c.id, "title": c.title, "description": c.description,
                 "requirements": c.requirements, "mandatory": c.mandatory}
                for c in self.clauses
            ],
            "controls": [
                {"id": c.id, "title": c.title, "description": c.description,
                 "objective": c.objective, "guidance": c.guidance}
                for c in self.controls
            ],
        }
