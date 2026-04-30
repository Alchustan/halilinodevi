import json

notebook_path = '/Users/baris/Projects/halilinodevi/odev_tr.ipynb'

replacements = {
    # Plot titles and labels
    "'Elbow Method For Optimal K'": "'Optimal K İçin Dirsek Yöntemi'",
    "'Number of Clusters (K)'": "'Küme Sayısı (K)'",
    "'Inertia'": "'Eylemsizlik'",
    "'Customer Distribution Across Segments'": "'Segmentlere Göre Müşteri Dağılımı'",
    "'Number of Customers'": "'Müşteri Sayısı'",
    "'2D PCA Visualization of Customer Segments'": "'Müşteri Segmentlerinin 2B PCA Görselleştirmesi'",
    "'Recency by Segment'": "'Segmente Göre Yenilik (Recency)'",
    "'Frequency by Segment'": "'Segmente Göre Sıklık (Frequency)'",
    "'Monetary by Segment'": "'Segmente Göre Parasal Değer (Monetary)'",
    "'Top 10 Words in Positive Reviews'": "'Pozitif Yorumlardaki En Sık 10 Kelime'",
    "'Top 10 Words in Negative Reviews'": "'Negatif Yorumlardaki En Sık 10 Kelime'",
    "'2D PCA Cluster Separation'": "'2B PCA Küme Ayrımı'",
    
    # Segment labels
    "'VIP Customers'": "'VIP Müşteriler'",
    "'Loyal Customers'": "'Sadık Müşteriler'",
    "'Lost Customers'": "'Kaybedilen Müşteriler'",
    "'New / At Risk'": "'Yeni / Risk Altında'",
    "'New / At Risk Customers'": "'Yeni / Risk Altındaki Müşteriler'",
    
    # Other print statements that might be visible in outputs (though output is not saved unless run)
    "\"Scaled Features (First 5 rows):\"": "\"Ölçeklenmiş Özellikler (İlk 5 Satır):\"",
    "\"Cluster Summary (Mean Values):\"": "\"Küme Özeti (Ortalama Değerler):\"",
    "\"Based on the elbow plot, the curve starts to flatten around K={optimal_k}.\"": "\"Dirsek grafiğine göre, eğri K={optimal_k} civarında düzleşmeye başlıyor.\""
}

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            modified_line = line
            for eng, tr in replacements.items():
                modified_line = modified_line.replace(eng, tr)
            new_source.append(modified_line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Görsel metinleri Türkçeye çevrildi.")
