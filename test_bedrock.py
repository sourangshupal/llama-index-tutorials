#!/usr/bin/env python3
"""
Test script for AWS Bedrock integration with LlamaIndex
This script tests both LLM and embedding functionality
"""

import os
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.core import Settings, VectorStoreIndex, Document

def test_bedrock_llm():
    """Test Bedrock LLM (Claude) integration"""
    print("=" * 60)
    print("TEST 1: Testing Bedrock LLM (Claude)")
    print("=" * 60)

    try:
        llm = BedrockConverse(
            model="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
            region_name="us-east-1",
            temperature=0.1,
            max_tokens=10000,
        )

        print("✓ BedrockConverse initialized successfully")
        print("\nSending test prompt: 'What is AI in one sentence?'")

        response = llm.complete("What is AI in one sentence?")
        print(f"\n✓ Response received:\n{response.text}\n")

        return True
    except Exception as e:
        print(f"\n✗ Error: {str(e)}\n")
        print("Possible issues:")
        print("  - AWS credentials not configured correctly")
        print("  - Model not enabled in AWS Bedrock Console")
        print("  - Incorrect region specified")
        return False


def test_bedrock_embeddings():
    """Test Bedrock Embeddings integration"""
    print("=" * 60)
    print("TEST 2: Testing Bedrock Embeddings")
    print("=" * 60)

    try:
        embed_model = BedrockEmbedding(
            model_name="amazon.titan-embed-text-v2:0",
            region_name="us-east-1",
        )

        print("✓ BedrockEmbedding initialized successfully")

        # Test single embedding
        test_text = "LlamaIndex is a data framework for LLM applications"
        print(f"\nGenerating embedding for: '{test_text}'")

        embedding = embed_model.get_text_embedding(test_text)
        print(f"✓ Embedding generated successfully")
        print(f"  Dimension: {len(embedding)}")
        print(f"  First 5 values: {embedding[:5]}\n")

        # Test batch embeddings
        test_batch = ["Hello", "World", "AI"]
        print(f"Generating batch embeddings for: {test_batch}")

        embeddings = embed_model.get_text_embedding_batch(test_batch)
        print(f"✓ Batch embeddings generated successfully")
        print(f"  Number of embeddings: {len(embeddings)}\n")

        return True
    except Exception as e:
        print(f"\n✗ Error: {str(e)}\n")
        print("Possible issues:")
        print("  - AWS credentials not configured correctly")
        print("  - Embedding model not enabled in AWS Bedrock Console")
        print("  - Incorrect model name or region")
        return False


def test_bedrock_rag():
    """Test end-to-end RAG with Bedrock"""
    print("=" * 60)
    print("TEST 3: Testing End-to-End RAG Pipeline")
    print("=" * 60)

    try:
        # Configure Settings
        Settings.llm = BedrockConverse(
            model="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
            region_name="us-east-1",
            temperature=0.1,
        )

        Settings.embed_model = BedrockEmbedding(
            model_name="amazon.titan-embed-text-v2:0",
            region_name="us-east-1",
        )

        print("✓ Settings configured with Bedrock LLM and Embeddings")

        # Create sample documents
        documents = [
            Document(text="LlamaIndex is a data framework for building LLM applications. It provides tools for data ingestion, indexing, and querying."),
            Document(text="AWS Bedrock is a fully managed service that offers foundation models from leading AI companies through a single API."),
            Document(text="Claude is an AI assistant created by Anthropic, known for being helpful, harmless, and honest."),
        ]

        print(f"\n✓ Created {len(documents)} sample documents")

        # Create index
        print("Building vector index...")
        index = VectorStoreIndex.from_documents(documents)
        print("✓ Vector index created successfully")

        # Create query engine
        query_engine = index.as_query_engine(similarity_top_k=2)
        print("✓ Query engine created")

        # Test query
        test_query = "What is LlamaIndex?"
        print(f"\nQuerying: '{test_query}'")

        response = query_engine.query(test_query)
        print(f"\n✓ Query executed successfully")
        print(f"\nResponse:\n{response}\n")

        return True
    except Exception as e:
        print(f"\n✗ Error: {str(e)}\n")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("AWS BEDROCK + LLAMAINDEX INTEGRATION TEST")
    print("=" * 60 + "\n")

    # Check AWS credentials
    print("Checking AWS credentials...")
    aws_profile = os.environ.get('AWS_PROFILE', 'default')
    aws_region = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
    print(f"  AWS Profile: {aws_profile}")
    print(f"  AWS Region: {aws_region}\n")

    results = []

    # Run tests
    results.append(("Bedrock LLM", test_bedrock_llm()))
    results.append(("Bedrock Embeddings", test_bedrock_embeddings()))
    results.append(("End-to-End RAG", test_bedrock_rag()))

    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(result[1] for result in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("Your AWS Bedrock integration is working correctly.")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("\nTroubleshooting:")
        print("1. Verify AWS credentials: aws sts get-caller-identity")
        print("2. Check Bedrock model access in AWS Console")
        print("3. Ensure you're in the correct region (us-east-1)")
        print("4. Install packages: pip install llama-index-llms-bedrock-converse llama-index-embeddings-bedrock boto3")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
