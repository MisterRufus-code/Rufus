"""Distributed task queue for Rufus pipeline jobs.

Requires Redis and Celery:
    pip install celery redis
    docker run -d -p 6379:6379 redis:alpine

See worker.py for usage.
"""
