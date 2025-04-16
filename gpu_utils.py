import logging
import torch
import os
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

class GPUManager:
    """
    Utility class to manage GPU resources for the bot
    """
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.gpu_available else "cpu")
        self.memory_allocated = 0
        
        # Log GPU information
        if self.gpu_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            logger.info(f"GPU acceleration enabled: {gpu_name}")
            logger.info(f"Number of GPUs available: {gpu_count}")
            logger.info(f"CUDA version: {torch.version.cuda}")
        else:
            logger.info("GPU acceleration not available, using CPU")
    
    def get_device(self) -> torch.device:
        """
        Get the current device (GPU or CPU)
        
        Returns:
            torch.device: The current device
        """
        return self.device
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get current GPU memory statistics
        
        Returns:
            Dict with memory statistics
        """
        if not self.gpu_available:
            return {"available": False, "allocated": 0, "reserved": 0, "total": 0}
        
        try:
            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_reserved(0)
            total = torch.cuda.get_device_properties(0).total_memory
            
            return {
                "available": True,
                "allocated": allocated / (1024 ** 2),  # MB
                "reserved": reserved / (1024 ** 2),    # MB
                "total": total / (1024 ** 2),          # MB
                "utilization": allocated / total * 100  # percentage
            }
        except Exception as e:
            logger.error(f"Error getting GPU memory stats: {e}")
            return {"available": True, "error": str(e)}
    
    def optimize_for_inference(self) -> None:
        """
        Apply optimizations for inference mode
        """
        if self.gpu_available:
            # Set to inference mode
            torch.set_grad_enabled(False)
            
            # Set memory optimization flags
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            
            # Enable TF32 precision if available (on NVIDIA Ampere GPUs)
            if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            # Enable cudnn benchmark mode for optimized performance
            torch.backends.cudnn.benchmark = True
            
            logger.info("Applied GPU optimizations for inference")
    
    def clear_cache(self) -> None:
        """
        Clear GPU cache to free up memory
        """
        if self.gpu_available:
            try:
                torch.cuda.empty_cache()
                logger.debug("Cleared GPU cache")
            except Exception as e:
                logger.error(f"Error clearing GPU cache: {e}")

# Create a singleton instance
gpu_manager = GPUManager()
