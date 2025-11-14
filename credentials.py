from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url="https://eb89f6f0-8fdb-4058-8275-46ef6116079a.us-west-2-0.aws.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.wwdOxPACBCgt1silDekXZ4WuAZ1vnKy2OFK_Ahbbjh8",
)

print(qdrant_client.get_collections())