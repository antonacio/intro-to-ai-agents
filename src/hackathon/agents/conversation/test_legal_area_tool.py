"""Test script to verify the classify_legal_area tool with enum enforcement."""

from hackathon.tools.legal_tools import classify_legal_area, LegalAreas
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os

print("🧪 TESTING LEGAL AREA CLASSIFICATION TOOL")
print("=" * 50)

# Test 1: Direct function call with enum
print("1. Testing direct function call with enum...")
try:
    result = classify_legal_area(LegalAreas.MERGERS_AND_ACQUISITIONS)
    print(f"   ✅ Success: {result['area_display_name']}")
    print(f"   Key questions: {len(result['key_questions'])}")
    print(f"   Focus areas: {result['focus_areas']}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Test with LLM tool calling (if API key available)
print("\n2. Testing with LLM tool calling...")
if not os.getenv("OPENAI_API_KEY"):
    print("   ⚠️  Skipping LLM test - OPENAI_API_KEY not set")
else:
    try:
        # Initialize LLM with the tool
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        llm_with_tools = llm.bind_tools([classify_legal_area])
        
        # Create a prompt that should trigger the tool
        system_prompt = """You are a legal intake specialist. When a client mentions M&A, mergers, acquisitions, or deal-making, you should use the classify_legal_area tool to get guidance on what questions to ask.

Available legal areas include: mergers_and_acquisitions, commercial_contracts, intellectual_property, data_protection, employment_dispute, regulatory_compliance, and others."""

        user_prompt = "The client mentioned they are acquiring another company and need legal help with the transaction."
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm_with_tools.invoke(messages)
        
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"   ✅ LLM called tool: {response.tool_calls[0]['name']}")
            print(f"   Tool args: {response.tool_calls[0]['args']}")
        else:
            print("   ⚠️  LLM did not call the tool")
            print(f"   Response: {response.content[:100]}...")
            
    except Exception as e:
        print(f"   ❌ LLM test error: {e}")

# Test 3: Show available enum values
print(f"\n3. Available legal areas:")
for area in LegalAreas:
    print(f"   - {area.value} ({area.value.replace('_', ' ').title()})")

print(f"\n🎯 Tool testing complete!")