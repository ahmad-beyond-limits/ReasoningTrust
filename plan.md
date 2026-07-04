1. Select **15 mathematical word problems** from the **GSM8K public dataset** from huggingface.

2. For **each** of the 10 problems, generate these **4 LLM responses**:

   * Correct answer + wrong reasoning chain.
   * Correct answer + correct reasoning.
   * Wrong answer + correct reasoning.
   * Wrong answer + wrong reasoning.

10 * 4 = 40 answer plus rating will be generated. in jsons. all independent now context or memeory.

3. Define **4 LLM judge personas**:

   * Math teacher.
   * Data scientist.
   * Policy maker.
   * General user.

4. Present **all 4 response conditions** for each problem to **each of the 4 judge personas**.

5. Do **not** tell the judge personas whether the answer is correct or incorrect.

6. Have each judge persona rate **trust in the answer** on a **1–7 scale**.

7. Have each judge persona rate **quality of the explanation** on a **1–7 scale**.

step 4,5,6,7 will require 40 * 4 = 160 prompts in total. will store in jsons and analyze later. All independent no context or memory.

Total api calls will be around 200. we will use groq qwen 3.6 and cerebras glm 4.6  for this.