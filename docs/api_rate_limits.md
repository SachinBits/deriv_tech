# API Rate Limits

Rate limits protect the platform and keep response times predictable for every customer. This page covers the REST API. Only the REST API is documented here.

## Limits by plan

The API rate limit for the Standard plan is 100 requests per minute. The API rate limit for the Enterprise plan is 1,000 requests per minute. Limits are measured over a rolling one-minute window.

## How limits are applied

Limits apply per API key. If an organisation uses several API keys, each key has its own allowance. Creating extra keys to work around the limit is not permitted under the terms of service.

## Exceeding the limit

Clients that exceed the API rate limit receive an HTTP 429 Too Many Requests response with a Retry-After header. The Retry-After header states how many seconds the client should wait before sending the next request. Clients should back off and retry after that interval instead of retrying immediately.

## Best practices

Cache responses that do not change often, batch work where the endpoint allows it, and spread scheduled jobs across the minute rather than starting them all at once.
