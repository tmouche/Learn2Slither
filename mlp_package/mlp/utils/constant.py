from typing import List

COLUMNS: List[str] = ["ID", "Diagnosis", 
		"mean radius", "std radius", "worst radius", 
		"mean texture", "std texture", "worst texture",
		"mean perimeter", "std perimeter", "worst perimeter",
		"mean area", "std area", "worst area",
		"mean smoothness", "std smoothness", "worst smoothness",
		"mean compactness", "std compactness", "worst compactness",
		"mean concavity", "std concavity", "worst concavity",
		"mean concave points", "std concave points", "worst concave points",
		"mean symmetry", "std symmetry", "worst symmetry",
		"mean fractal dimension", "std fractal dimension", "worst fractal dimension"
	]

DROP_COLUMNS: List[str] = ["ID",
            "mean texture", "mean area", "mean compactness", "mean concavity", "mean symmetry",
            "std radius", "std texture", "std area", "std smoothness", "std compactness", "std concavity", "std symmetry", "std fractal dimension",
            "worst texture", "worst perimeter", "worst area", "worst smoothness", "worst compactness", "worst concavity", "worst symmetry", "worst fractal dimension"
        ]

SPLIT_DATASET: int = 90

POSITIV = [1, 0]

FOLDER_NET_SAVE = "network_save"

