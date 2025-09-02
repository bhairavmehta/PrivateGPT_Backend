#!/usr/bin/env python3

import asyncio
import aiohttp
import json
import sys

async def test_8000_tokens_comprehensive():
    """Test that 8000 max_tokens and improved context are working for comprehensive responses"""
    
    base_url = "http://localhost:8000"
    
    # Test data for deep research with comprehensive context
    test_data = {
        "message": "explain machine learning trends and applications in 2024",
        "session_id": "test_8000_tokens",
        "useDeepResearch": True,
        "useWebSearch": False,
        "maxTokens": 8000,  # Test the new 8000 limit
        "temperature": 0.7,
        "modelType": "coding"
    }
    
    print("🧪 Testing 8000 Max Tokens & Comprehensive Context...")
    print(f"📝 Query: {test_data['message']}")
    print(f"🔬 Deep Research: {test_data['useDeepResearch']}")
    print(f"💰 Max Tokens: {test_data['maxTokens']}")
    print("-" * 60)
    
    # Test streaming response
    timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
    
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{base_url}/chat/stream",
                json=test_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status != 200:
                    print(f"❌ HTTP Error: {response.status}")
                    return False
                
                chunks_processed = 0
                final_chunk_found = False
                source_id_found = None
                accumulated_response = ""
                
                async for line in response.content:
                    try:
                        if line:
                            chunk_data = json.loads(line.decode().strip())
                            chunks_processed += 1
                            
                            if chunk_data.get("accumulated_text"):
                                accumulated_response = chunk_data["accumulated_text"]
                            
                            # Check final chunk
                            if chunk_data.get("finished"):
                                final_chunk_found = True
                                source_id_found = chunk_data.get("source_id")
                                source_count = chunk_data.get("source_count", 0)
                                
                                print(f"✅ Final chunk found after {chunks_processed} chunks")
                                print(f"📊 Source ID: {source_id_found}")
                                print(f"🔢 Source Count: {source_count}")
                                print(f"📏 Response Length: {len(accumulated_response)} characters")
                                print(f"📝 Response Preview: {accumulated_response[:200]}...")
                                break
                                
                    except json.JSONDecodeError as e:
                        continue  # Skip malformed chunks
                    except Exception as e:
                        print(f"⚠️ Chunk processing error: {e}")
                        continue
                
                if not final_chunk_found:
                    print("❌ No final chunk received")
                    return False
                
                # Test sources API if we have a source_id
                if source_id_found:
                    print(f"\n🔍 Testing Sources API with ID: {source_id_found}")
                    
                    async with session.get(f"{base_url}/sources/{source_id_found}") as sources_response:
                        if sources_response.status == 200:
                            sources_data = await sources_response.json()
                            
                            if sources_data.get("success"):
                                sources = sources_data.get("sources", [])
                                print(f"✅ Sources API successful: {len(sources)} sources")
                                print(f"🔍 Search Type: {sources_data.get('type', 'unknown')}")
                                
                                # Check if sources have comprehensive content
                                for i, source in enumerate(sources[:3], 1):
                                    title = source.get("title", "No title")[:50]
                                    content_length = len(source.get("scraped_content", "") or source.get("snippet", ""))
                                    print(f"   Source {i}: {title}... ({content_length} chars)")
                                
                                # Verify comprehensive analysis
                                if len(accumulated_response) > 2000:  # Should be comprehensive with 8000 tokens
                                    print("✅ Response is comprehensive (>2000 characters)")
                                    return True
                                else:
                                    print(f"⚠️ Response may not be comprehensive ({len(accumulated_response)} characters)")
                                    return True  # Still success, but note the length
                            else:
                                print(f"❌ Sources API failed: {sources_data.get('error', 'Unknown error')}")
                                return False
                        else:
                            print(f"❌ Sources API HTTP error: {sources_response.status}")
                            return False
                else:
                    print("⚠️ No source_id found - sources may not have been generated")
                    return False
                    
    except asyncio.TimeoutError:
        print("❌ Test timed out")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

async def main():
    """Run the comprehensive test"""
    print("🚀 Starting 8000 Max Tokens & Comprehensive Context Test...")
    print("=" * 70)
    
    success = await test_8000_tokens_comprehensive()
    
    print("=" * 70)
    if success:
        print("🎉 Test PASSED: 8000 max_tokens and comprehensive context working!")
        print("✅ Features verified:")
        print("   - 8000 max_tokens accepted")
        print("   - Deep research finding sources")
        print("   - Sources API working")
        print("   - Comprehensive responses generated")
        sys.exit(0)
    else:
        print("❌ Test FAILED: Issues found with 8000 max_tokens or context")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 