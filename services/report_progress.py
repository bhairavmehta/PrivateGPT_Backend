import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProgressPhase:
    """A phase in the report generation process"""
    name: str
    description: str
    total_steps: int
    current_step: int = 0
    status: str = "pending"  # pending, running, completed, error
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None

@dataclass
class ReportProgress:
    """Complete progress information for a report generation"""
    report_id: str
    status: str  # queued, running, completed, error, cancelled
    overall_progress: float = 0.0
    current_phase: str = ""
    phases: List[ProgressPhase] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    model_calls_made: int = 0
    total_model_calls_planned: int = 50
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.phases is None:
            self.phases = []

class ReportProgressTracker:
    """Service for tracking report generation progress across multiple concurrent requests"""
    
    def __init__(self):
        self.progress_data: Dict[str, ReportProgress] = {}
        self.cleanup_interval = 3600  # Clean up completed reports after 1 hour
        
        # Define the standard phases for ultra-comprehensive reports
        self.standard_phases = [
            ProgressPhase("initialization", "🚀 Initializing ultra-comprehensive report generation", 2),
            ProgressPhase("content_analysis", "🧠 Deep Multi-Layer Content Analysis", 8),
            ProgressPhase("architecture_design", "🏗️ AI-Driven Report Architecture Design", 3),
            ProgressPhase("research", "🔬 Autonomous Multi-Source Research", 5),
            ProgressPhase("analytics", "📊 Advanced Data Analytics & Pattern Recognition", 6),
            ProgressPhase("visualizations", "🎨 Beautiful Complex Visualization Creation", 5),
            ProgressPhase("content_generation", "✍️ Multi-Pass Content Generation & Refinement", 15),
            ProgressPhase("quality_enhancement", "🔍 AI Quality Enhancement & Validation", 3),
            ProgressPhase("document_assembly", "📋 Professional Document Assembly", 3)
        ]
        
    def create_report_progress(self, report_type: str = "ultra_comprehensive") -> str:
        """Create a new report progress tracker and return its ID"""
        report_id = str(uuid.uuid4())[:8]
        
        progress = ReportProgress(
            report_id=report_id,
            status="queued",
            start_time=datetime.now(),
            phases=[phase for phase in self.standard_phases],
            total_model_calls_planned=50
        )
        
        self.progress_data[report_id] = progress
        logger.info(f"Created progress tracker for report {report_id}")
        
        return report_id
    
    def update_overall_status(self, report_id: str, status: str, error_message: str = None):
        """Update the overall status of a report"""
        if report_id not in self.progress_data:
            logger.warning(f"Report {report_id} not found in progress tracker")
            return
            
        progress = self.progress_data[report_id]
        progress.status = status
        
        if status == "running" and not progress.start_time:
            progress.start_time = datetime.now()
        elif status in ["completed", "error", "cancelled"]:
            progress.end_time = datetime.now()
            
        if error_message:
            progress.error_message = error_message
            
        # Calculate overall progress
        progress.overall_progress = self._calculate_overall_progress(progress)
        
        logger.info(f"Updated report {report_id} status to {status}")
    
    def start_phase(self, report_id: str, phase_name: str):
        """Start a specific phase"""
        if report_id not in self.progress_data:
            return
            
        progress = self.progress_data[report_id]
        
        # Find the phase
        for phase in progress.phases:
            if phase.name == phase_name:
                phase.status = "running"
                phase.start_time = datetime.now()
                phase.current_step = 0
                progress.current_phase = phase.description
                break
                
        progress.overall_progress = self._calculate_overall_progress(progress)
        logger.info(f"Started phase {phase_name} for report {report_id}")
    
    def update_phase_step(self, report_id: str, phase_name: str, step: int, description: str = None):
        """Update the current step within a phase"""
        if report_id not in self.progress_data:
            return
            
        progress = self.progress_data[report_id]
        
        # Find the phase
        for phase in progress.phases:
            if phase.name == phase_name:
                phase.current_step = min(step, phase.total_steps)
                if description:
                    progress.current_phase = description
                break
                
        progress.overall_progress = self._calculate_overall_progress(progress)
    
    def complete_phase(self, report_id: str, phase_name: str):
        """Mark a phase as completed"""
        if report_id not in self.progress_data:
            return
            
        progress = self.progress_data[report_id]
        
        # Find the phase
        for phase in progress.phases:
            if phase.name == phase_name:
                phase.status = "completed"
                phase.end_time = datetime.now()
                phase.current_step = phase.total_steps
                break
                
        progress.overall_progress = self._calculate_overall_progress(progress)
        logger.info(f"Completed phase {phase_name} for report {report_id}")
    
    def error_phase(self, report_id: str, phase_name: str, error_message: str):
        """Mark a phase as having an error"""
        if report_id not in self.progress_data:
            return
            
        progress = self.progress_data[report_id]
        
        # Find the phase
        for phase in progress.phases:
            if phase.name == phase_name:
                phase.status = "error"
                phase.end_time = datetime.now()
                phase.error_message = error_message
                break
                
        progress.status = "error"
        progress.error_message = error_message
        progress.overall_progress = self._calculate_overall_progress(progress)
    
    def update_model_calls(self, report_id: str, model_calls_made: int):
        """Update the number of model calls made"""
        if report_id not in self.progress_data:
            return
            
        progress = self.progress_data[report_id]
        progress.model_calls_made = model_calls_made
    
    def set_result(self, report_id: str, result: Dict[str, Any]):
        """Set the final result for a completed report"""
        if report_id not in self.progress_data:
            return
            
        progress = self.progress_data[report_id]
        progress.result = result
        progress.status = "completed"
        progress.end_time = datetime.now()
        progress.overall_progress = 100.0
        
        # Mark all phases as completed
        for phase in progress.phases:
            if phase.status != "completed":
                phase.status = "completed"
                phase.current_step = phase.total_steps
                
        logger.info(f"Set result for completed report {report_id}")
    
    def get_progress(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Get the current progress for a report"""
        if report_id not in self.progress_data:
            return None
            
        progress = self.progress_data[report_id]
        
        # Convert to dictionary for API response
        result = asdict(progress)
        
        # Add timing information
        if progress.start_time:
            result["elapsed_time"] = (datetime.now() - progress.start_time).total_seconds()
            
        # Add estimated completion time
        if progress.overall_progress > 10 and progress.status == "running":
            elapsed = (datetime.now() - progress.start_time).total_seconds()
            estimated_total = (elapsed / progress.overall_progress) * 100
            estimated_remaining = estimated_total - elapsed
            result["estimated_remaining_seconds"] = estimated_remaining
            
        return result
    
    def _calculate_overall_progress(self, progress: ReportProgress) -> float:
        """Calculate overall progress based on phase completion"""
        if not progress.phases:
            return 0.0
            
        total_steps = sum(phase.total_steps for phase in progress.phases)
        completed_steps = 0
        
        for phase in progress.phases:
            if phase.status == "completed":
                completed_steps += phase.total_steps
            elif phase.status == "running":
                completed_steps += phase.current_step
                
        return min(100.0, (completed_steps / total_steps) * 100) if total_steps > 0 else 0.0
    
    def list_active_reports(self) -> List[Dict[str, Any]]:
        """List all active (non-completed) reports"""
        active_reports = []
        
        for report_id, progress in self.progress_data.items():
            if progress.status in ["queued", "running"]:
                progress_dict = asdict(progress)
                if progress.start_time:
                    progress_dict["elapsed_time"] = (datetime.now() - progress.start_time).total_seconds()
                active_reports.append(progress_dict)
                
        return active_reports
    
    def cleanup_old_reports(self):
        """Clean up old completed reports"""
        cutoff_time = datetime.now().timestamp() - self.cleanup_interval
        
        to_remove = []
        for report_id, progress in self.progress_data.items():
            if (progress.status in ["completed", "error", "cancelled"] and 
                progress.end_time and 
                progress.end_time.timestamp() < cutoff_time):
                to_remove.append(report_id)
                
        for report_id in to_remove:
            del self.progress_data[report_id]
            logger.info(f"Cleaned up old report {report_id}")
    
    def cancel_report(self, report_id: str) -> bool:
        """Cancel a report generation"""
        if report_id not in self.progress_data:
            return False
            
        progress = self.progress_data[report_id]
        if progress.status in ["completed", "error"]:
            return False
            
        progress.status = "cancelled"
        progress.end_time = datetime.now()
        
        # Mark current running phase as cancelled
        for phase in progress.phases:
            if phase.status == "running":
                phase.status = "cancelled"
                phase.end_time = datetime.now()
                break
                
        logger.info(f"Cancelled report {report_id}")
        return True

# Global instance
progress_tracker = ReportProgressTracker() 