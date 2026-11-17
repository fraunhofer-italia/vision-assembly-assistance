#############################################################################
##### PURPOSE : ------------------ ##########################################
##### Real operator assistance must:                        #################
##### 1) Give recovery suggestions                          #################
##### 2) Allow retry                                        #################
##### 3) Prevent unsafe progression                         #################
#############################################################################


class RecoveryManager:

    def __init__(self):
        pass

    #------------------------------------------------------------------------

    def generate_feedback(self, stage, final_score, tool_valid, part_valid=True):

        feedback = {
            "action": "RETRY_STAGE",
            "reason": []
        }

        if not tool_valid:
            feedback["reason"].append(
                f"Incorrect tool. Expected {stage['expected_tools']}"
            )

        if not part_valid:
            feedback["reason"].append(
                f"Incorrect active part. Expected {stage.get('active_part_candidates', stage.get('expected_parts', []))}"
            )

        if final_score < stage["score_threshold"]:
            feedback["reason"].append(
                f"Motion quality below threshold ({stage['score_threshold']})"
            )

        return feedback
