from dataclasses import dataclass


@dataclass(slots=True)
class RuntimeMetrics:
    chunks_received: int = 0
    chunks_dropped: int = 0
    silent_chunks: int = 0
    non_voiced_chunks: int = 0
    clipped_chunks: int = 0
    feature_frames_processed: int = 0
    invalid_inference_outputs: int = 0
    low_confidence_matches: int = 0
    accepted_matches: int = 0
    slide_triggers_sent: int = 0
    manual_override_suppressed_triggers: int = 0
    osc_send_failures: int = 0
    queue_high_water_mark: int = 0
