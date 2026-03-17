"""NIST AI Risk Management Framework (AI RMF 1.0) encoding.

Provides the four core functions, 19 categories, and 72 subcategories
of the NIST AI RMF with obligation types and implementation guidance.
"""

from dataclasses import dataclass, field
from enum import Enum
import json


class RMFFunction(Enum):
    """The four core functions of the NIST AI RMF."""
    GOVERN = "GOVERN"
    MAP = "MAP"
    MEASURE = "MEASURE"
    MANAGE = "MANAGE"


@dataclass
class Category:
    """A category within an RMF function."""
    id: str
    function: str
    title: str
    description: str
    subcategories: list = field(default_factory=list)


@dataclass
class Subcategory:
    """A subcategory with specific guidance."""
    id: str
    category_id: str
    title: str
    description: str
    obligation_type: str
    suggested_actions: list = field(default_factory=list)


class NISTRiskManagementFramework:
    """NIST Artificial Intelligence Risk Management Framework (AI RMF 1.0).

    Released January 26, 2023. Voluntary framework intended to improve the
    ability to incorporate trustworthiness considerations into the design,
    development, use, and evaluation of AI products, services, and systems.
    """

    def __init__(self):
        self.functions = self._init_functions()
        self.categories = self._init_categories()
        self.subcategories = self._init_subcategories()

    def _init_functions(self) -> dict:
        """Define the four core RMF functions."""
        return {
            "GOVERN": {
                "id": "GOVERN",
                "name": "Govern",
                "description": (
                    "Cultivate and implement a culture of risk management within "
                    "organizations designing, developing, deploying, or using AI systems. "
                    "Establish policies, processes, procedures, and practices across the "
                    "organization related to the mapping, measuring, and managing of AI risks."
                ),
                "outcomes": [
                    "Policies, processes, procedures and practices are in place and enforced",
                    "Accountability structures are in place",
                    "Workforce diversity, equity, inclusion and accessibility are prioritized",
                    "Organizational teams are committed to a culture of risk management",
                    "Processes for engagement with relevant AI actors are defined and followed",
                    "Policies and procedures address AI risks arising in third-party contexts",
                ],
            },
            "MAP": {
                "id": "MAP",
                "name": "Map",
                "description": (
                    "Establish context to frame risks related to an AI system. "
                    "Identify and assess the full scope of risks, benefits, and costs "
                    "to individuals, organizations, and society."
                ),
                "outcomes": [
                    "Context is established and understood",
                    "Categorization of the AI system is performed",
                    "AI capabilities, targeted usage, goals, and expected benefits are understood",
                    "Risks and benefits are mapped for all components of the AI system",
                    "Impacts to individuals, groups, communities, organizations and society are characterized",
                ],
            },
            "MEASURE": {
                "id": "MEASURE",
                "name": "Measure",
                "description": (
                    "Employ quantitative, qualitative, or mixed-method tools, techniques, "
                    "and methodologies to analyze, assess, benchmark, and monitor AI risk "
                    "and related impacts."
                ),
                "outcomes": [
                    "Appropriate methods and metrics are identified and applied",
                    "AI systems are evaluated for trustworthy characteristics",
                    "Mechanisms for tracking identified AI risks over time are in place",
                    "Feedback about efficacy of measurement is collected and integrated",
                ],
            },
            "MANAGE": {
                "id": "MANAGE",
                "name": "Manage",
                "description": (
                    "Allocate resources to mapped and measured risks on a regular basis "
                    "and as defined by the GOVERN function. Plan, resource, and execute "
                    "risk response strategies."
                ),
                "outcomes": [
                    "AI risks based on assessments and other analytical output are prioritized, responded to, and managed",
                    "Strategies to maximize AI benefits and minimize negative impacts are planned and prepared",
                    "AI risks and benefits from third-party resources are managed",
                    "Risk treatments including response and recovery and communication plans are in place and monitored",
                ],
            },
        }

    def _init_categories(self) -> list:
        """Define the 19 categories across four functions."""
        return [
            # GOVERN categories (6)
            Category("GOVERN 1", "GOVERN", "Policies, processes, procedures, and practices",
                     "Policies, processes, procedures, and practices across the organization related to the mapping, measuring, and managing of AI risks are in place, transparent, and implemented effectively."),
            Category("GOVERN 2", "GOVERN", "Accountability",
                     "Accountability structures are in place so that the appropriate teams and individuals are empowered, responsible, and trained for mapping, measuring, and managing AI risks."),
            Category("GOVERN 3", "GOVERN", "Workforce diversity, equity, inclusion, and accessibility",
                     "Workforce diversity, equity, inclusion, and accessibility processes are prioritized in the mapping, measuring, and managing of AI risks throughout the lifecycle."),
            Category("GOVERN 4", "GOVERN", "Organizational teams and culture",
                     "Organizational teams are committed to a culture that considers and communicates AI risk."),
            Category("GOVERN 5", "GOVERN", "Engagement with AI actors",
                     "Processes are in place for robust engagement with relevant AI actors."),
            Category("GOVERN 6", "GOVERN", "Third-party risk management",
                     "Policies and procedures are in place to address AI risks and benefits arising from third-party software and data and other supply chain issues."),

            # MAP categories (5)
            Category("MAP 1", "MAP", "Context and intended purpose",
                     "Context is established and understood. The intended purpose, setting, potential benefits, and potential costs are well-defined."),
            Category("MAP 2", "MAP", "Categorization and classification",
                     "Categorization of the AI system is performed including assessment of whether the system is an AI system."),
            Category("MAP 3", "MAP", "Benefits and costs",
                     "AI capabilities, targeted usage, goals, expected benefits and costs compared with appropriate benchmarks are understood."),
            Category("MAP 4", "MAP", "Risk identification",
                     "Risks and benefits are mapped for all components of the AI system including third-party software and data."),
            Category("MAP 5", "MAP", "Impact assessment",
                     "Impacts to individuals, groups, communities, organizations, and society are characterized."),

            # MEASURE categories (4)
            Category("MEASURE 1", "MEASURE", "Appropriate methods and metrics",
                     "Appropriate methods and metrics are identified and applied."),
            Category("MEASURE 2", "MEASURE", "AI system evaluation",
                     "AI systems are evaluated for trustworthy characteristics."),
            Category("MEASURE 3", "MEASURE", "Risk tracking",
                     "Mechanisms for tracking identified AI risks over time are in place."),
            Category("MEASURE 4", "MEASURE", "Feedback collection",
                     "Feedback about efficacy of measurement is collected and integrated."),

            # MANAGE categories (4)
            Category("MANAGE 1", "MANAGE", "Risk prioritization and response",
                     "AI risks based on assessments and other analytical output from the MAP and MEASURE functions are prioritized, responded to, and managed."),
            Category("MANAGE 2", "MANAGE", "Strategy planning",
                     "Strategies to maximize AI benefits and minimize negative impacts are planned, prepared, implemented, documented, and informed by input from relevant AI actors."),
            Category("MANAGE 3", "MANAGE", "Third-party risk management",
                     "AI risks and benefits from third-party resources are managed."),
            Category("MANAGE 4", "MANAGE", "Risk treatment and communication",
                     "Risk treatments, including response and recovery, and communication plans for the identified and measured AI risks are documented and monitored regularly."),
        ]

    def _init_subcategories(self) -> list:
        """Define all 72 subcategories with descriptions and suggested actions."""
        subs = []

        # GOVERN 1 subcategories (1.1-1.7)
        gov1 = [
            ("GOVERN 1.1", "Legal and regulatory requirements involving AI are understood, managed, and documented.",
             ["Identify applicable laws and regulations", "Document compliance approach", "Monitor for regulatory changes"]),
            ("GOVERN 1.2", "The characteristics of trustworthy AI are integrated into organizational policies, processes, and procedures.",
             ["Define trustworthiness criteria", "Embed in development standards", "Include in procurement requirements"]),
            ("GOVERN 1.3", "Processes, procedures, and practices are in place to determine the needed level of risk management activities based on the assessed risk.",
             ["Develop risk tiering approach", "Proportionate oversight based on risk level", "Document thresholds and criteria"]),
            ("GOVERN 1.4", "The risk management process and its outcomes are established through transparent policies and procedures and are regularly reviewed.",
             ["Publish risk management policies", "Conduct periodic reviews", "Ensure stakeholder accessibility"]),
            ("GOVERN 1.5", "Ongoing monitoring and periodic review of the risk management process and its outcomes are planned, organizationally resourced, and informing updates.",
             ["Allocate resources for monitoring", "Schedule periodic reviews", "Incorporate lessons learned"]),
            ("GOVERN 1.6", "Mechanisms are in place to inventory AI systems and are resourced for regular use.",
             ["Create AI system inventory", "Track deployment and retirement", "Resource inventory maintenance"]),
            ("GOVERN 1.7", "Processes and procedures for decommissioning and phasing out AI systems are defined and documented.",
             ["Define decommission criteria", "Document data retention policies", "Plan stakeholder notification"]),
        ]
        for sub_id, desc, actions in gov1:
            subs.append(Subcategory(sub_id, "GOVERN 1", sub_id, desc, "recommended", actions))

        # GOVERN 2 subcategories (2.1-2.3)
        gov2 = [
            ("GOVERN 2.1", "Roles and responsibilities and lines of communication related to mapping, measuring, and managing AI risks are documented and clearly communicated.",
             ["Define RACI matrix", "Document communication channels", "Train responsible individuals"]),
            ("GOVERN 2.2", "The organization's personnel and partners receive AI risk management training to enable them to perform their duties and responsibilities.",
             ["Develop training curricula", "Provide regular refresher training", "Track training completion"]),
            ("GOVERN 2.3", "Executive leadership of the organization takes responsibility for decisions about risks associated with AI system development and deployment.",
             ["Assign executive sponsorship", "Include AI risk in board reporting", "Ensure leadership sign-off"]),
        ]
        for sub_id, desc, actions in gov2:
            subs.append(Subcategory(sub_id, "GOVERN 2", sub_id, desc, "recommended", actions))

        # GOVERN 3 subcategories (3.1-3.2)
        gov3 = [
            ("GOVERN 3.1", "Decision-making related to mapping, measuring, and managing AI risks throughout the lifecycle is informed by a diverse team.",
             ["Include diverse perspectives", "Seek external review where needed", "Document representation"]),
            ("GOVERN 3.2", "Policies and procedures are in place to define and differentiate roles and responsibilities for human-AI interaction and oversight.",
             ["Define human-AI interaction boundaries", "Specify override capabilities", "Document escalation procedures"]),
        ]
        for sub_id, desc, actions in gov3:
            subs.append(Subcategory(sub_id, "GOVERN 3", sub_id, desc, "recommended", actions))

        # GOVERN 4 subcategories (4.1-4.3)
        gov4 = [
            ("GOVERN 4.1", "Organizational practices are in place to foster a critical thinking and safety-first mindset in the design, development, and deployment of AI systems.",
             ["Encourage challenge culture", "Reward safety-conscious behaviour", "Conduct retrospectives"]),
            ("GOVERN 4.2", "Organizational teams document the risks and potential impacts of the AI technology they design, develop, deploy, and monitor.",
             ["Maintain risk registers", "Link risks to specific systems", "Report on risk evolution"]),
            ("GOVERN 4.3", "Organizational practices are in place to enable testing, identification of incidents, and information sharing.",
             ["Establish testing protocols", "Create incident reporting channels", "Participate in industry sharing"]),
        ]
        for sub_id, desc, actions in gov4:
            subs.append(Subcategory(sub_id, "GOVERN 4", sub_id, desc, "recommended", actions))

        # GOVERN 5 subcategories (5.1-5.2)
        gov5 = [
            ("GOVERN 5.1", "Organizational policies and practices are in place to collect, consider, prioritize, and integrate feedback from those external to the team.",
             ["Create feedback channels", "Process and prioritize feedback", "Close the loop with respondents"]),
            ("GOVERN 5.2", "Mechanisms are established to enable AI actors to regularly incorporate adjudicated feedback from relevant interested parties.",
             ["Adjudicate conflicting feedback", "Track feedback integration", "Report on feedback outcomes"]),
        ]
        for sub_id, desc, actions in gov5:
            subs.append(Subcategory(sub_id, "GOVERN 5", sub_id, desc, "recommended", actions))

        # GOVERN 6 subcategories (6.1-6.2)
        gov6 = [
            ("GOVERN 6.1", "Policies and procedures are in place that address AI risks associated with third-party entities.",
             ["Assess third-party AI risks", "Include AI provisions in contracts", "Monitor third-party compliance"]),
            ("GOVERN 6.2", "Contingency processes are in place for addressing AI system failures or incidents from third-party entities.",
             ["Define incident response for third-party failures", "Establish SLAs", "Plan for vendor transitions"]),
        ]
        for sub_id, desc, actions in gov6:
            subs.append(Subcategory(sub_id, "GOVERN 6", sub_id, desc, "recommended", actions))

        # MAP 1 subcategories (1.1-1.6)
        map1 = [
            ("MAP 1.1", "Intended purpose, potentially beneficial uses, context of use, and users of the AI system are understood and documented.",
             ["Document intended use cases", "Identify user groups", "Specify deployment context"]),
            ("MAP 1.2", "Interdisciplinary AI actors, competencies, skills, and capacities needed for achieving identified system goals are identified.",
             ["Map required competencies", "Identify skill gaps", "Plan for interdisciplinary teams"]),
            ("MAP 1.3", "The organization's mission and relevant goals for the AI system are understood and documented.",
             ["Align AI with organizational mission", "Document AI goals", "Link to strategic objectives"]),
            ("MAP 1.4", "The business value or context of business use has been clearly defined or the AI system is not being developed.",
             ["Define business value proposition", "Establish success metrics", "Conduct feasibility assessment"]),
            ("MAP 1.5", "Organizational risk tolerance is determined and clearly communicated.",
             ["Define risk appetite statement", "Communicate thresholds", "Calibrate across organization"]),
            ("MAP 1.6", "System requirements including those related to trustworthy AI are elicited from and understood by relevant AI actors.",
             ["Gather requirements from stakeholders", "Include trustworthiness requirements", "Validate with AI actors"]),
        ]
        for sub_id, desc, actions in map1:
            subs.append(Subcategory(sub_id, "MAP 1", sub_id, desc, "recommended", actions))

        # MAP 2 subcategories (2.1-2.3)
        map2 = [
            ("MAP 2.1", "The specific task, and methods used to implement the task, that the AI system will support is defined.",
             ["Specify tasks supported by AI", "Document AI methods used", "Identify alternatives"]),
            ("MAP 2.2", "Information about the AI system's knowledge limits and how the system behaves in unanticipated conditions is documented.",
             ["Document known limitations", "Characterize edge case behaviour", "Identify failure modes"]),
            ("MAP 2.3", "Scientific integrity and TEVV (Test, Evaluation, Validation, and Verification) considerations are identified and documented.",
             ["Plan TEVV strategy", "Identify scientific integrity measures", "Document validation approach"]),
        ]
        for sub_id, desc, actions in map2:
            subs.append(Subcategory(sub_id, "MAP 2", sub_id, desc, "recommended", actions))

        # MAP 3 subcategories (3.1-3.5)
        map3 = [
            ("MAP 3.1", "Potential benefits of intended functionality are examined and documented.",
             ["Enumerate benefits", "Quantify where possible", "Validate with stakeholders"]),
            ("MAP 3.2", "Potential costs of the AI system including risks of non-deployment are examined and documented.",
             ["Assess deployment costs", "Evaluate non-deployment risks", "Compare alternatives"]),
            ("MAP 3.3", "Options for tradeoffs and alternatives to the AI system are documented.",
             ["Identify non-AI alternatives", "Document tradeoff analysis", "Present options to decision-makers"]),
            ("MAP 3.4", "Measurable performance targets are defined for the AI system.",
             ["Set quantitative thresholds", "Define acceptance criteria", "Align with business goals"]),
            ("MAP 3.5", "The AI system to be deployed is demonstrated to perform as intended for the target population and use context.",
             ["Conduct user acceptance testing", "Validate with representative data", "Document performance results"]),
        ]
        for sub_id, desc, actions in map3:
            subs.append(Subcategory(sub_id, "MAP 3", sub_id, desc, "recommended", actions))

        # MAP 4 subcategories (4.1-4.2)
        map4 = [
            ("MAP 4.1", "Approaches for mapping AI risks encompass socio-technical dimensions and are built into every stage of the AI lifecycle.",
             ["Integrate socio-technical risk assessment", "Apply at each lifecycle stage", "Include diverse perspectives"]),
            ("MAP 4.2", "Internal risk controls for components of the AI system including third-party AI resources are identified and documented.",
             ["Catalog internal controls", "Assess third-party controls", "Document gaps"]),
        ]
        for sub_id, desc, actions in map4:
            subs.append(Subcategory(sub_id, "MAP 4", sub_id, desc, "recommended", actions))

        # MAP 5 subcategories (5.1-5.2)
        map5 = [
            ("MAP 5.1", "Impacts to individuals, groups, communities, organizations, and society are characterized.",
             ["Identify affected populations", "Assess impact severity and likelihood", "Document impact pathways"]),
            ("MAP 5.2", "Practices and personnel for defining, understanding, and documenting AI system impacts are in place.",
             ["Assign impact assessment responsibility", "Train on impact methodology", "Conduct periodic reassessment"]),
        ]
        for sub_id, desc, actions in map5:
            subs.append(Subcategory(sub_id, "MAP 5", sub_id, desc, "recommended", actions))

        # MEASURE 1 subcategories (1.1-1.3)
        meas1 = [
            ("MEASURE 1.1", "Approaches and metrics for measurement of AI risks are selected based on established state-of-the-art or standardized methodologies.",
             ["Survey applicable metrics", "Adopt validated methodologies", "Document metric rationale"]),
            ("MEASURE 1.2", "Appropriateness of AI metrics and effectiveness of existing measures are regularly assessed.",
             ["Review metric relevance", "Assess measurement effectiveness", "Update as needed"]),
            ("MEASURE 1.3", "Internal processes for assessing and evaluating AI system performance, fairness, and other trustworthy characteristics are in place.",
             ["Establish evaluation protocols", "Include fairness assessments", "Document and track results"]),
        ]
        for sub_id, desc, actions in meas1:
            subs.append(Subcategory(sub_id, "MEASURE 1", sub_id, desc, "recommended", actions))

        # MEASURE 2 subcategories (2.1-2.13)
        meas2_items = [
            ("MEASURE 2.1", "Test sets, metrics, and details about the tools used during evaluation are documented."),
            ("MEASURE 2.2", "Evaluations involving human subjects meet ethical requirements."),
            ("MEASURE 2.3", "AI system performance or assurance criteria are measured qualitatively or quantitatively and demonstrated for conditions similar to deployment setting."),
            ("MEASURE 2.4", "The AI system's functionality and behavior in deployment is monitored against the system requirements."),
            ("MEASURE 2.5", "The AI system's fairness and bias, as appropriate, are evaluated and results are documented."),
            ("MEASURE 2.6", "The AI system is evaluated for reliability, security, resilience and robustness in conditions of expected and unexpected input and adversarial conditions."),
            ("MEASURE 2.7", "AI system security and resilience are evaluated and results documented, as are risks to privacy and rights."),
            ("MEASURE 2.8", "Risks associated with transparency and accountability are evaluated."),
            ("MEASURE 2.9", "The AI model is explained, validated, and interpreted, and outcomes documented for each trustworthy AI characteristic."),
            ("MEASURE 2.10", "Privacy risk of the AI system is examined and documented."),
            ("MEASURE 2.11", "Fairness and bias, both in AI system performance and underlying data, are evaluated and documented."),
            ("MEASURE 2.12", "Environmental impact and sustainability of AI model training and management activities are assessed and documented."),
            ("MEASURE 2.13", "Effectiveness of the employed TEVV metrics is evaluated and documented."),
        ]
        for sub_id, desc in meas2_items:
            subs.append(Subcategory(sub_id, "MEASURE 2", sub_id, desc, "recommended",
                                    ["Conduct specified evaluation", "Document results", "Integrate findings into risk management"]))

        # MEASURE 3 subcategories (3.1-3.3)
        meas3 = [
            ("MEASURE 3.1", "Approaches, personnel, and documentation are in place to regularly identify and track existing, unanticipated, and emergent risks."),
            ("MEASURE 3.2", "Risk tracking approaches are considered for settings where AI systems are operated by, or interact with, humans."),
            ("MEASURE 3.3", "Feedback processes for end users and impacted communities regarding the quality of system performance are established and integrated."),
        ]
        for sub_id, desc in meas3:
            subs.append(Subcategory(sub_id, "MEASURE 3", sub_id, desc, "recommended",
                                    ["Implement tracking mechanism", "Include human interaction contexts", "Establish feedback loops"]))

        # MEASURE 4 subcategories (4.1-4.3)
        meas4 = [
            ("MEASURE 4.1", "Measurement approaches for identifying AI risks are connected to deployment context(s) and informed by domain expertise."),
            ("MEASURE 4.2", "Measurement results regarding AI system trustworthiness in deployment context(s) and target communities are informed by domain experts."),
            ("MEASURE 4.3", "Outcomes of pre-deployment testing and in-deployment monitoring are used to update risk management processes."),
        ]
        for sub_id, desc in meas4:
            subs.append(Subcategory(sub_id, "MEASURE 4", sub_id, desc, "recommended",
                                    ["Connect to deployment context", "Involve domain experts", "Feed into risk management updates"]))

        # MANAGE 1 subcategories (1.1-1.4)
        man1 = [
            ("MANAGE 1.1", "A determination is made as to whether the AI system achieves its intended purpose and stated objectives."),
            ("MANAGE 1.2", "Treatment of documented AI risks is prioritized based on impact, likelihood, and the available resources."),
            ("MANAGE 1.3", "Responses to the AI risks deemed high priority are developed, planned, and documented."),
            ("MANAGE 1.4", "Negative residual risks (remaining after mitigation) are documented for communication."),
        ]
        for sub_id, desc in man1:
            subs.append(Subcategory(sub_id, "MANAGE 1", sub_id, desc, "recommended",
                                    ["Assess against intended purpose", "Prioritize risk treatments", "Document residual risk"]))

        # MANAGE 2 subcategories (2.1-2.4)
        man2 = [
            ("MANAGE 2.1", "Resources required to manage AI risks are taken into account along with viable nonAI alternative systems, approaches, or methods."),
            ("MANAGE 2.2", "Mechanisms are in place and applied to sustain the value of deployed AI systems."),
            ("MANAGE 2.3", "Procedures are followed to respond to and recover from a previously unknown risk when it is identified."),
            ("MANAGE 2.4", "Mechanisms are in place and applied to document incidents, communicate them to relevant parties and address them."),
        ]
        for sub_id, desc in man2:
            subs.append(Subcategory(sub_id, "MANAGE 2", sub_id, desc, "recommended",
                                    ["Resource risk management activities", "Sustain deployed system value", "Handle incidents and unknowns"]))

        # MANAGE 3 subcategories (3.1-3.2)
        man3 = [
            ("MANAGE 3.1", "AI risks and benefits from third-party resources are regularly monitored, and risk controls are applied and documented."),
            ("MANAGE 3.2", "Pre-trained models used for development are monitored as part of AI system regular monitoring and maintenance."),
        ]
        for sub_id, desc in man3:
            subs.append(Subcategory(sub_id, "MANAGE 3", sub_id, desc, "recommended",
                                    ["Monitor third-party resources", "Apply risk controls", "Document monitoring results"]))

        # MANAGE 4 subcategories (4.1-4.3)
        man4 = [
            ("MANAGE 4.1", "Post-deployment AI system monitoring plans are implemented and include mechanisms for capturing and evaluating input from users and affected communities."),
            ("MANAGE 4.2", "Measurable activities for continual improvements are integrated into AI system updates and include regular engagement with interested parties."),
            ("MANAGE 4.3", "Incidents and errors are communicated to relevant AI actors including affected communities."),
        ]
        for sub_id, desc in man4:
            subs.append(Subcategory(sub_id, "MANAGE 4", sub_id, desc, "recommended",
                                    ["Implement post-deployment monitoring", "Integrate continual improvement", "Communicate incidents"]))

        return subs

    def get_function(self, func_id: str) -> dict:
        """Get a function definition by ID."""
        return self.functions.get(func_id.upper())

    def get_category(self, cat_id: str) -> Category:
        """Get a category by ID (e.g., 'GOVERN 1')."""
        for cat in self.categories:
            if cat.id == cat_id:
                return cat
        return None

    def get_subcategory(self, sub_id: str) -> Subcategory:
        """Get a subcategory by ID (e.g., 'GOVERN 1.1')."""
        for sub in self.subcategories:
            if sub.id == sub_id:
                return sub
        return None

    def get_by_function(self, function: str) -> list:
        """Get all categories for a given function."""
        func_upper = function.upper()
        return [c for c in self.categories if c.function == func_upper]

    def get_all_subcategories(self) -> list:
        """Get all 72 subcategories."""
        return list(self.subcategories)

    def map_to_eu_ai_act(self) -> dict:
        """Rough mapping of NIST RMF categories to EU AI Act articles."""
        return {
            "GOVERN 1": {"eu_articles": ["Article 17"], "concept": "Quality management system"},
            "GOVERN 2": {"eu_articles": ["Article 16", "Article 26"], "concept": "Provider/deployer responsibilities"},
            "GOVERN 3": {"eu_articles": ["Article 10"], "concept": "Data governance and bias prevention"},
            "GOVERN 4": {"eu_articles": ["Article 4"], "concept": "AI literacy and culture"},
            "GOVERN 5": {"eu_articles": ["Article 86"], "concept": "Stakeholder engagement and explanation"},
            "GOVERN 6": {"eu_articles": ["Article 25"], "concept": "Value chain responsibilities"},
            "MAP 1": {"eu_articles": ["Article 9", "Article 6"], "concept": "Risk assessment and classification"},
            "MAP 2": {"eu_articles": ["Article 11"], "concept": "Technical documentation"},
            "MAP 3": {"eu_articles": ["Article 9"], "concept": "Risk management system"},
            "MAP 4": {"eu_articles": ["Article 9", "Article 10"], "concept": "Risk identification and data governance"},
            "MAP 5": {"eu_articles": ["Article 27"], "concept": "Fundamental rights impact assessment"},
            "MEASURE 1": {"eu_articles": ["Article 15"], "concept": "Accuracy, robustness, cybersecurity"},
            "MEASURE 2": {"eu_articles": ["Article 9", "Article 15"], "concept": "System evaluation and testing"},
            "MEASURE 3": {"eu_articles": ["Article 72"], "concept": "Post-market monitoring"},
            "MEASURE 4": {"eu_articles": ["Article 72", "Article 73"], "concept": "Deployment monitoring and incident reporting"},
            "MANAGE 1": {"eu_articles": ["Article 9"], "concept": "Risk treatment and prioritization"},
            "MANAGE 2": {"eu_articles": ["Article 16", "Article 73"], "concept": "Corrective action and incident handling"},
            "MANAGE 3": {"eu_articles": ["Article 25"], "concept": "Third-party and supply chain management"},
            "MANAGE 4": {"eu_articles": ["Article 72", "Article 73"], "concept": "Post-market monitoring and incident communication"},
        }

    def get_implementation_tiers(self) -> dict:
        """Get the 4 implementation tiers for organizational maturity."""
        return {
            "Tier 1 — Partial": {
                "description": "Risk management is ad hoc and reactive. Limited awareness of AI risks at organizational level.",
                "characteristics": [
                    "No formal AI risk management process",
                    "Ad hoc risk identification",
                    "Limited organizational awareness",
                    "Minimal external engagement",
                ],
            },
            "Tier 2 — Risk-Informed": {
                "description": "Risk management practices are approved by management but may not be organization-wide.",
                "characteristics": [
                    "Some risk management practices exist",
                    "Management-approved but inconsistently applied",
                    "Awareness exists but processes are not integrated",
                    "Some external engagement",
                ],
            },
            "Tier 3 — Repeatable": {
                "description": "Organizational risk management practices are formally established and regularly updated.",
                "characteristics": [
                    "Formal policies and procedures in place",
                    "Consistently applied across the organization",
                    "Regular updates based on lessons learned",
                    "Active external engagement",
                ],
            },
            "Tier 4 — Adaptive": {
                "description": "Organization adapts practices based on lessons learned and predictive indicators.",
                "characteristics": [
                    "Continuous improvement processes",
                    "Proactive and predictive risk management",
                    "Fully integrated into organizational culture",
                    "Leading external engagement and information sharing",
                ],
            },
        }

    def get_ai_rmf_profiles(self) -> list:
        """Get example AI RMF profiles for common use cases."""
        return [
            {
                "name": "High-Risk AI System Profile",
                "description": "Profile for AI systems with significant potential impact on individuals",
                "priority_functions": ["GOVERN", "MAP", "MEASURE", "MANAGE"],
                "priority_categories": [
                    "GOVERN 1", "GOVERN 2", "MAP 1", "MAP 5",
                    "MEASURE 1", "MEASURE 2", "MANAGE 1", "MANAGE 4",
                ],
                "minimum_tier": "Tier 3 — Repeatable",
                "emphasis": "Comprehensive risk assessment, continuous monitoring, robust accountability",
            },
            {
                "name": "General-Purpose AI Profile",
                "description": "Profile for general-purpose AI including large language models",
                "priority_functions": ["GOVERN", "MAP", "MEASURE"],
                "priority_categories": [
                    "GOVERN 1", "GOVERN 4", "GOVERN 6",
                    "MAP 1", "MAP 2", "MAP 3",
                    "MEASURE 2", "MEASURE 3",
                ],
                "minimum_tier": "Tier 2 — Risk-Informed",
                "emphasis": "Broad risk mapping, transparency, third-party risk management",
            },
            {
                "name": "Low-Risk AI System Profile",
                "description": "Profile for AI systems with minimal potential for harm",
                "priority_functions": ["GOVERN", "MAP"],
                "priority_categories": [
                    "GOVERN 1", "MAP 1", "MAP 2",
                ],
                "minimum_tier": "Tier 1 — Partial",
                "emphasis": "Basic documentation and context establishment",
            },
        ]

    def summary(self) -> str:
        """Human-readable summary of the NIST AI RMF encoding."""
        lines = [
            "NIST AI Risk Management Framework (AI RMF 1.0)",
            f"  Functions: {len(self.functions)}",
            f"  Categories: {len(self.categories)}",
            f"  Subcategories: {len(self.subcategories)}",
            "  Framework type: Voluntary",
            "  Released: January 26, 2023",
            "  Functions:",
        ]
        for func_id, func in self.functions.items():
            cats = self.get_by_function(func_id)
            n_subs = sum(1 for s in self.subcategories if s.category_id in [c.id for c in cats])
            lines.append(f"    {func_id}: {func['name']} ({len(cats)} categories, {n_subs} subcategories)")
        return "\n".join(lines)

    def to_json(self) -> dict:
        """Serialize the framework to JSON-compatible dict."""
        return {
            "framework": "nist_ai_rmf",
            "name": "NIST AI Risk Management Framework 1.0",
            "version": "1.0",
            "released": "2023-01-26",
            "type": "voluntary",
            "functions": self.functions,
            "categories": [
                {"id": c.id, "function": c.function, "title": c.title, "description": c.description}
                for c in self.categories
            ],
            "subcategories": [
                {"id": s.id, "category_id": s.category_id, "title": s.title,
                 "description": s.description, "obligation_type": s.obligation_type,
                 "suggested_actions": s.suggested_actions}
                for s in self.subcategories
            ],
        }
