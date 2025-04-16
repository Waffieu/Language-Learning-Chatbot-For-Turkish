import logging
import time
import asyncio
import os
import sys
from memory import Memory
from gpu_utils import gpu_manager
import google.generativeai as genai
import config

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize Gemini
genai.configure(api_key=config.GEMINI_API_KEY)

async def test_memory_performance():
    """Test memory system performance"""
    logger.info("Testing memory system performance...")
    
    # Create a memory instance
    memory = Memory()
    
    # Test chat IDs
    test_chat_ids = [1001, 1002, 1003, 1004, 1005]
    
    # Add messages to each chat
    start_time = time.time()
    for chat_id in test_chat_ids:
        for i in range(50):  # Add 50 messages per chat
            memory.add_message(chat_id, "user" if i % 2 == 0 else "model", f"Test message {i}")
    
    add_time = time.time() - start_time
    logger.info(f"Added 250 messages to 5 chats in {add_time:.4f} seconds")
    
    # Test retrieval performance
    start_time = time.time()
    for chat_id in test_chat_ids:
        # Get short memory 10 times
        for _ in range(10):
            short_memory = memory.get_short_memory(chat_id)
        
        # Get long memory 5 times
        for _ in range(5):
            long_memory = memory.get_long_memory(chat_id)
    
    retrieval_time = time.time() - start_time
    logger.info(f"Retrieved memory 75 times in {retrieval_time:.4f} seconds")
    
    # Test auto-save
    logger.info("Testing auto-save functionality...")
    await asyncio.sleep(config.MEMORY_AUTOSAVE_INTERVAL + 5)
    
    # Verify files were created
    for chat_id in test_chat_ids:
        memory_file = os.path.join(config.MEMORY_DIR, f"memory_{chat_id}.json")
        if os.path.exists(memory_file):
            logger.info(f"Memory file for chat {chat_id} was auto-saved successfully")
        else:
            logger.error(f"Memory file for chat {chat_id} was not auto-saved")
    
    # Clean up test files
    for chat_id in test_chat_ids:
        memory_file = os.path.join(config.MEMORY_DIR, f"memory_{chat_id}.json")
        if os.path.exists(memory_file):
            os.remove(memory_file)
    
    # Shutdown memory system
    memory.shutdown()
    logger.info("Memory performance test completed")

async def test_gpu_integration():
    """Test GPU integration with Gemini API"""
    logger.info("Testing GPU integration...")
    
    # Check if GPU is available
    if gpu_manager.gpu_available:
        logger.info(f"GPU is available: {gpu_manager.get_device()}")
        gpu_stats = gpu_manager.get_memory_stats()
        logger.info(f"GPU memory: {gpu_stats['allocated']:.2f}MB allocated, {gpu_stats['total']:.2f}MB total")
    else:
        logger.info("GPU is not available, using CPU")
    
    # Apply optimizations
    gpu_manager.optimize_for_inference()
    
    # Test Gemini API with GPU optimizations
    logger.info("Testing Gemini API with GPU optimizations...")
    
    # Configure Gemini model
    model = genai.GenerativeModel(
        model_name=config.GEMINI_MODEL,
        generation_config={
            "temperature": config.GEMINI_TEMPERATURE,
            "top_p": config.GEMINI_TOP_P,
            "top_k": config.GEMINI_TOP_K,
            "max_output_tokens": config.GEMINI_MAX_OUTPUT_TOKENS,
        },
        safety_settings=config.SAFETY_SETTINGS
    )
    
    # Test prompt
    prompt = "Write a short poem about a fox with two tails who is a mechanical genius."
    
    # Generate response with timing
    start_time = time.time()
    try:
        response = await asyncio.to_thread(
            lambda: model.generate_content(prompt).text
        )
        end_time = time.time()
        
        logger.info(f"Response generated in {end_time - start_time:.2f} seconds")
        logger.info(f"Response: {response}")
        
        # Check GPU memory after generation
        if gpu_manager.gpu_available:
            gpu_stats = gpu_manager.get_memory_stats()
            logger.info(f"GPU memory after generation: {gpu_stats['allocated']:.2f}MB allocated")
            
            # Test cache clearing
            gpu_manager.clear_cache()
            gpu_stats = gpu_manager.get_memory_stats()
            logger.info(f"GPU memory after cache clearing: {gpu_stats['allocated']:.2f}MB allocated")
    
    except Exception as e:
        logger.error(f"Error generating response: {e}")
    
    logger.info("GPU integration test completed")

async def main():
    """Run all tests"""
    logger.info("Starting optimization tests...")
    
    # Test memory performance
    await test_memory_performance()
    
    # Test GPU integration
    await test_gpu_integration()
    
    logger.info("All tests completed")

if __name__ == "__main__":
    asyncio.run(main())
