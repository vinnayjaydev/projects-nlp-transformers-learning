from sentence_transformers import util

def route_query(
    query,
    embedding_model,
    router_embeddings,
):
    query_embedding = embedding_model.encode(
        query,
        convert_to_tensor=True,
    )

    scores = {}

    for department, embedding in router_embeddings.items():

        similarity = util.cos_sim(
            query_embedding,
            embedding,
        )

        scores[department] = float(similarity)

    best_department = max(
        scores,
        key=scores.get,
    )

    return best_department, scores