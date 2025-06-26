from enum import Enum, unique
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from hackathon.tools.rag import retrieve as base_retrieve


@unique
class LegalAreas(str, Enum):
    MERGERS_AND_ACQUISITIONS = "mergers_and_acquisitions"
    CORPORATE_GOVERNANCE = "corporate_governance"
    PRIVATE_EQUITY_AND_VENTURE_CAPITAL = "private_equity_and_venture_capital"
    PERSONAL_INJURY = "personal_injury"
    MEDICAL_NEGLIGENCE = "medical_negligence"
    EMPLOYMENT_DISPUTE = "employment_dispute"
    COMMERCIAL_CONTRACTS = "commercial_contracts"
    PROPERTY_DISPUTE = "property_dispute"
    FAMILY_LAW = "family_law"
    CRIMINAL_DEFENSE = "criminal_defense"
    IMMIGRATION = "immigration"
    WILLS_AND_PROBATE = "wills_and_probate"
    PROFESSIONAL_NEGLIGENCE = "professional_negligence"
    FINANCIAL_DISPUTES = "financial_disputes"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    DATA_PROTECTION = "data_protection"
    INSURANCE_CLAIMS = "insurance_claims"
    CONSTRUCTION_DISPUTES = "construction_disputes"
    DEFAMATION_AND_REPUTATION = "defamation_and_reputation"
    LICENSING_AND_PERMITS = "licensing_and_permits"
    TENANCY_EVICTION = "tenancy_eviction"
    JUDICIAL_REVIEW = "judicial_review"


@tool(parse_docstring=True)
def search_web(query: str, max_results: int = 3) -> str:
    """Search the web for information about legal topics or general information.

    Args:
        query: The search query to look up
        max_results: The maximum number of results to return (capped at 5)

    Returns:
        str: Search results from the web
    """
    # Check if Tavily API key is available
    if not os.getenv("TAVILY_API_KEY"):
        return "Web search is not available. Please set TAVILY_API_KEY environment variable."

    if max_results > 5:
        max_results = 5

    try:
        # Initialize Tavily search
        search = TavilySearchResults(max_results=max_results)
        results = search.run(query)

        # Format results
        if isinstance(results, list):
            formatted_results = []
            for result in results:
                formatted_results.append(
                    f"Title: {result.get('title', 'N/A')}\n"
                    f"URL: {result.get('url', 'N/A')}\n"
                    f"Content: {result.get('content', 'N/A')}\n"
                )
            return "\n---\n".join(formatted_results)
        else:
            return str(results)
    except Exception as e:
        return f"Error performing web search: {str(e)}"


@tool(parse_docstring=True)
def classify_legal_area(legal_area: LegalAreas) -> dict:
    """Set the legal area classification and get area-specific questioning guidance.
    
    Call this tool when you've determined the client's primary legal area to get
    specific guidance on what information to gather for that area.

    Args:
        legal_area: The legal area from the LegalAreas enum

    Returns:
        dict: Classification result with area-specific questioning guidance
    """
    
    # Mapping of legal areas to specific questioning guidance
    area_guidance = {
        "mergers_and_acquisitions": {
            "key_questions": [
                "What type of transaction is this? (merger, acquisition, joint venture, etc.)",
                "What is the approximate deal size/valuation?",
                "What stage is the transaction in?",
                "Are there any regulatory concerns or approvals needed?",
                "What is the target timeline for completion?"
            ],
            "focus_areas": ["Due diligence", "Valuation", "Regulatory compliance", "Deal structure", "Financing"],
            "urgency_indicators": ["LOI signed", "Due diligence phase", "Regulatory deadlines"]
        },
        "commercial_contracts": {
            "key_questions": [
                "What type of contract needs review/drafting?",
                "Who are the parties involved?",
                "What is the contract value and duration?",
                "Are there specific terms or clauses of concern?",
                "What are the key deliverables and timelines?"
            ],
            "focus_areas": ["Terms and conditions", "Risk allocation", "Payment terms", "Termination clauses", "Dispute resolution"],
            "urgency_indicators": ["Contract deadline", "Negotiation phase", "Signature required"]
        },
        "intellectual_property": {
            "key_questions": [
                "What type of IP is involved? (patents, trademarks, copyrights, trade secrets)",
                "Is this for protection, enforcement, or defense?",
                "Are there any pending deadlines or filing requirements?",
                "Is there potential infringement involved?",
                "What is the business value of the IP?"
            ],
            "focus_areas": ["Patent prosecution", "Trademark registration", "IP licensing", "Infringement analysis", "IP portfolio strategy"],
            "urgency_indicators": ["Filing deadlines", "Infringement claims", "Product launch timing"]
        },
        "data_protection": {
            "key_questions": [
                "What type of data is involved? (personal, sensitive, customer data)",
                "Which regulations apply? (GDPR, CCPA, HIPAA, etc.)",
                "Is this for compliance review or incident response?",
                "What are your data processing activities?",
                "Have there been any data incidents or breaches?"
            ],
            "focus_areas": ["Compliance assessment", "Privacy policies", "Data mapping", "Breach response", "International transfers"],
            "urgency_indicators": ["Regulatory deadlines", "Data breach", "Product launch", "Audit requirements"]
        },
        "employment_dispute": {
            "key_questions": [
                "What type of employment issue is this?",
                "How many employees are affected?",
                "Are there any pending claims or tribunals?",
                "What are the specific allegations or concerns?",
                "What is the desired outcome?"
            ],
            "focus_areas": ["Employment contracts", "Disciplinary procedures", "Discrimination claims", "Redundancy", "Settlement negotiations"],
            "urgency_indicators": ["Tribunal deadlines", "Employee grievances", "Termination proceedings"]
        },
        "regulatory_compliance": {
            "key_questions": [
                "Which industry regulations apply?",
                "What is the scope of compliance review needed?",
                "Are there any pending regulatory actions?",
                "What are the compliance deadlines?",
                "Have there been any regulatory communications?"
            ],
            "focus_areas": ["Regulatory framework", "Compliance procedures", "Risk assessment", "Regulatory reporting", "Stakeholder engagement"],
            "urgency_indicators": ["Regulatory deadlines", "Inspection notices", "Compliance breaches"]
        }
    }
    
    # Default guidance for areas not specifically mapped
    default_guidance = {
        "key_questions": [
            "What is the specific legal challenge or opportunity?",
            "What are the key stakeholders involved?",
            "What is the desired timeline for resolution?",
            "Are there any immediate deadlines or constraints?",
            "What is the potential business impact?"
        ],
        "focus_areas": ["Legal analysis", "Risk assessment", "Strategic planning", "Documentation", "Implementation"],
        "urgency_indicators": ["Legal deadlines", "Business milestones", "Regulatory requirements"]
    }
    
    # Convert enum to string value for lookup
    legal_area_value = legal_area.value
    
    # Get guidance for the specific area
    guidance = area_guidance.get(legal_area_value, default_guidance)
    
    return {
        "legal_area": legal_area_value,
        "status": "classified",
        "area_display_name": legal_area_value.replace('_', ' ').title(),
        "key_questions": guidance["key_questions"],
        "focus_areas": guidance["focus_areas"],
        "urgency_indicators": guidance["urgency_indicators"],
        "guidance": f"Focus on gathering information about: {', '.join(guidance['focus_areas'])}. Pay attention to urgency indicators like: {', '.join(guidance['urgency_indicators'])}."
    }


@tool(parse_docstring=True)
def end_conversation() -> dict:
    """End the conversation and prepare handoff to the deck drafting agent.

    Call this tool when you have gathered sufficient information from the client
    and are ready to conclude the intake consultation.

    Returns:
        dict: Handoff information for the deck drafting agent
    """
    return {
        "status": "conversation_ended",
        "handoff_to": "deck_drafting_agent",
        "message": "Perfect! I have all the information I need. I'm now passing this on to the relevant parties who specialize in your legal area. They will be getting back to you with a tailored pitch deck that showcases our expertise and proposed approach for your specific needs. You can expect to hear from them within 24-48 hours. Thank you for your time!"
    }


@tool(parse_docstring=True)
def extract_client_info(conversation_history: str) -> dict:
    """Extract relevant client information from the conversation history.

    Args:
        conversation_history: The conversation history with the client

    Returns:
        dict: Extracted client information
    """
    # TODO: Implement information extraction from conversation
    # Extract: company name, industry, size, specific needs, timeline, budget, etc.
    return {
        "company_name": "TODO",
        "industry": "TODO",
        "company_size": "TODO",
        "legal_needs": "TODO",
        "timeline": "TODO",
        "additional_info": {},
    }


@tool(parse_docstring=True)
def search_lawyers_db(legal_area: str, additional_filters: dict = None) -> list:
    """Search the lawyers database for experts in specific legal areas.

    Args:
        legal_area: The area of legal expertise required
        additional_filters: Additional filters like location, seniority, etc.

    Returns:
        list: List of lawyers matching the criteria
    """
    # TODO: Implement database search for lawyers
    # This will query your lawyers database with expertise, availability, etc.
    return [
        {
            "name": "TODO: Lawyer Name",
            "email": "TODO@example.com",
            "expertise": ["TODO: Area 1", "TODO: Area 2"],
            "experience_years": 0,
            "bio": "TODO: Lawyer bio",
        }
    ]


@tool(parse_docstring=True)
def retrieve_uploaded_docs(query: str) -> str:
    """Retrieve information from uploaded legal documents.

    This tool searches through documents that clients have uploaded
    to find relevant information for their legal needs.

    Args:
        query: The search query to look for in uploaded documents

    Returns:
        str: Relevant excerpts from uploaded documents
    """
    # Use the base retrieve function but add legal-specific context
    result = base_retrieve(query)

    # If result is a tuple (content, artifacts), extract the content
    if isinstance(result, tuple):
        content, _ = result
        return f"From uploaded legal documents:\n{content}"
    else:
        return f"From uploaded legal documents:\n{result}"
