# ADR-002: Use bounded queues for pipeline backpressure

## Status

Accepted

## Decision

Audio and presentation commands will cross worker boundaries through bounded queues. Producers must not block real-time audio callbacks.

## Consequences

Queue overflow is observable and handled explicitly, normally by dropping the oldest audio chunk rather than blocking capture.
