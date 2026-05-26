from langgraph.store.memory import InMemoryStore, IndexConfig

short_term_memory_store = InMemoryStore() # for text only
long_term_memory_store = InMemoryStore() # for text only

# short_term_memory_store = InMemoryStore(
#     index=IndexConfig(
#         dims=1536,
#         embed="text-embedding-3-small"
#     )
# )

# long_term_memory_store = InMemoryStore(
#     index=IndexConfig(
#         dims=1536,
#         embed="text-embedding-3-small"
#     )
# )