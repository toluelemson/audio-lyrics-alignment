# ADR-001: Use a modular monolith

## Status

Accepted

## Decision

Implement the MVP as one deployable Python process with explicit domain, application, port, and adapter boundaries.

## Rationale

The live pipeline requires low latency, bounded in-memory queues, and coordinated lifecycle management. Microservices would introduce unnecessary network latency and operational complexity before the core alignment approach is validated.
