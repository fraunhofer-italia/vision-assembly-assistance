SYNTHETIC_STAGE_CONFIG = [
    {
        "name": "StageA_circular",
        "expected_tools": ["NONE", "ScrewdriverS"],
        "score_threshold": 0.7,
        "reference_paths": [
            "data/references/stageA_exp1.npy",
            "data/references/stageA_exp2.npy"
        ]
        
    },
    {
        "name": "StageB_Linear",
        "expected_tools": ["ScrewdriverL"],
        "score_threshold": 0.75,
        "reference_paths": [
            "data/references/stageB_exp1.npy",
            "data/references/stageB_exp2.npy"
        ]
        
    },
    {
        "name": "StageC_Sinusoidal",
        "expected_tools": ["NONE","ScrewdriverL"],
        "reference_paths": "data/references/stage_C.npy",
        "score_threshold": 0.7
    }
]