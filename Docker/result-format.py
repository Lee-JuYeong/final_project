def format_results(data, cosine_sim):
    filtered_results = []
    for idx, row in data.iterrows():
        card_name = row['Card Name']
        sorted_indices = cosine_sim[idx].argsort()[::-1]
        filtered_similarities = [
            {'Similar Card': data.iloc[i]['Card Name'], 'Cosine Similarity': cosine_sim[idx][i]}
            for i in sorted_indices if data.iloc[i]['Card Name'] != card_name
        ]
        top_similar = filtered_similarities[:1]
        for similar in top_similar:
            filtered_results.append({
                'Card Name': card_name,
                'Similar Card': similar['Similar Card'],
                'Cosine Similarity': similar['Cosine Similarity']
            })
    return filtered_results
