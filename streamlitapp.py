import streamlit as st
import random
from datetime import datetime, timedelta
from typing import List, Dict
import json
import os
import pandas as pd

# --- File for storing submissions ---
SUBMISSIONS_FILE = "submissions.json"

class LegacyJavaCodeChallenge:
    def __init__(self):
        self.code_samples = self._load_java_code_samples()
        # Evaluation criteria weights based on difficulty
        self.evaluation_criteria = {
            'Easy': {'correct_requirements': 0.4, 'identified_bugs': 0.6},
            'Medium': {'correct_requirements': 0.3, 'identified_bugs': 0.5, 'code_improvements': 0.2},
            'Hard': {'correct_requirements': 0.2, 'identified_bugs': 0.5, 'code_improvements': 0.3}
        }
        # Max scores for each difficulty level (sum to 100)
        self.max_scores = {'Easy': 30, 'Medium': 30, 'Hard': 40}

    def _load_java_code_samples(self) -> List[Dict]:
        """Load various buggy Java code samples with hidden requirements"""
        return [
            {
                'id': 0,
                'name': "Order Processor Bug Fix",
                'difficulty': "Easy",
                'code': """
public class OrderProcessor {
    public static void main(String[] args) {
        int[] itemPrices = {100, 200, -50, 300};
        double discount = 0.1;

        int total = 0;
        for (int i = 0; i <= itemPrices.length; i++) {
            total += itemPrices[i];
        }

        if (total > 500) {
            total = (int) (total * discount);
        }

        System.out.println("Final Total: " + total);
    }
}
                """,
                'hints': [
                    "Check array bounds carefully",
                    "Consider validation of input values",
                    "Verify discount calculation logic",
                    "Think about edge cases for pricing"
                ],
                'expected_bugs': [
                    "ArrayIndexOutOfBoundsException due to incorrect loop condition (i <= itemPrices.length)",
                    "Negative prices are included in total calculation",
                    "Discount calculation is incorrect (multiplies total instead of subtracting discount)",
                    "No validation for negative prices"
                ],
                'expected_requirements': [
                    "Process all valid items in the order",
                    "Calculate total amount correctly",
                    "Apply discount when total exceeds threshold",
                    "Display final total to user"
                ]
            },
            {
                'id': 1,
                'name': "Minimum Keypad Presses Calculator",
                'difficulty': "Medium",
                'code': """
import java.util.*;

public class Solution {
    public int minimumPushes(String word) {
        // Step 1: Frequency map
        int[] freq = new int[26];
        for (char c : word.toCharArray()) {
            freq[c - 'a']--;
        }

        // Step 2: Sort characters by frequency (descending)
        List<Integer> frequencies = new ArrayList<>();
        for (int f : freq) {
            if (f >= 0) continue;
            frequencies.add(f);
        }

        frequencies.sort((a, b) -> a - b);

        // Step 3: Assign letters to positions on keys greedily
        int totalCost = 0;
        int index = 1;
        int maxPositions = 9 * 3;

        for (int i = 0; i <= frequencies.size(); i++) {
            int presses = (index / 8) + 1;
            totalCost += frequencies.get(i) * presses;
            index++;
        }

        return totalCost;
    }
}
                """,
                'hints': [
                    "Check frequency counting logic",
                    "Verify sorting order",
                    "Examine loop bounds carefully",
                    "Check key position calculation"
                ],
                'expected_bugs': [
                    "Incorrect frequency counting (decrement instead of increment)",
                    "Wrong condition for filtering frequencies (skips valid frequencies)",
                    "Incorrect sorting order (ascending instead of descending)",
                    "Wrong starting index for press calculation",
                    "Incorrect max positions calculation (9 keys instead of 8)",
                    "Array index out of bounds in final loop"
                ],
                'expected_requirements': [
                    "Calculate minimum key presses for given word",
                    "Optimize key assignment based on letter frequency",
                    "Handle all lowercase English letters",
                    "Return correct total presses count"
                ],
                'expected_improvements': [
                    "Add input validation for non-letter characters",
                    "Optimize memory usage for frequency storage",
                    "Add comments explaining the algorithm",
                    "Handle empty string input"
                ]
            },
            {
                'id': 2,
                'name': "String Scramble Checker",
                'difficulty': "Hard",
                'code': """
import java.util.*; // Added import for Map/HashMap

class Solution {
    Map<String, Boolean> map = new HashMap<>(); // Map key should be String

    public boolean isScramble(String s1, String s2) {
        int n = s1.length();
        
        // Add early exit condition if lengths are different
        if (n != s2.length()) {
            return false;
        }

        if (s1.equals(s2)) { // Use .equals() for string comparison
            return true;
        }

        // Use a unique string key for memoization
        String memoKey = s1 + "#" + s2; 
        if (map.containsKey(memoKey)) {
            return map.get(memoKey);
        }

        // Character frequency check
        int[] charCount = new int[26];
        for (int k = 0; k < n; k++) {
            charCount[s1.charAt(k) - 'a']++;
            charCount[s2.charAt(k) - 'a']--;
        }
        for (int count : charCount) {
            if (count != 0) {
                map.put(memoKey, false);
                return false; // Characters don't match
            }
        }

        for (int i = 1; i < n; i++) { // Loop condition changed: i < n
            // Scenario 1: No swap at partition
            if (isScramble(s1.substring(0, i), s2.substring(0, i)) &&
                isScramble(s1.substring(i), s2.substring(i))) {
                map.put(memoKey, true);
                return true;
            }

            // Scenario 2: Swap at partition
            if (isScramble(s1.substring(0, i), s2.substring(n - i)) &&
                isScramble(s1.substring(i), s2.substring(0, n - i))) {
                map.put(memoKey, true);
                return true;
            }
        }

        map.put(memoKey, false);
        return false;
    }
}
                """,
                'hints': [
                    "Check map key type and usage",
                    "Verify string comparison method",
                    "Examine array bounds carefully",
                    "Check array comparison method",
                    "Verify memoization key generation"
                ],
                'expected_bugs': [
                    "Incorrect map key type (Integer instead of String)",
                    "Using == for string comparison instead of equals()",
                    "Frequency array size incorrect (27 instead of 26)",
                    "Array index out of bounds in character access",
                    "Incorrect array comparison with ==",
                    "Using hashCode() for memoization key which can collide"
                ],
                'expected_requirements': [
                    "Determine if s2 is a scrambled version of s1",
                    "Use memoization to optimize performance",
                    "Handle all lowercase English letters",
                    "Return correct boolean result"
                ],
                'expected_improvements': [
                    "Add length check optimization at start",
                    "Add character frequency check optimization",
                    "Improve memoization key generation",
                    "Add input validation",
                    "Add comments explaining the algorithm"
                ]
            }
        ]

    def get_challenge_by_difficulty(self, difficulty: str) -> Dict:
        """Get challenge by difficulty level"""
        # Finds the first challenge with the given difficulty and returns it.
        # If you wanted to offer multiple challenges per difficulty, you'd
        # need to modify this to pick a random one, or cycle through them.
        for challenge in self.code_samples:
            if challenge['difficulty'] == difficulty:
                return challenge
        return None

    def evaluate_response(self, challenge_id: int,
                          requirements: List[str],
                          bugs: List[str],
                          improvements: List[str] = None) -> Dict: # Change return type to Dict
        """Evaluate candidate's response based on difficulty-specific criteria"""
        challenge = next(c for c in self.code_samples if c['id'] == challenge_id)
        difficulty = challenge['difficulty']
        criteria = self.evaluation_criteria[difficulty]
        max_score = self.max_scores[difficulty]

        # Detailed tracking for feedback
        identified_req_correct = [req for req in requirements if req in challenge['expected_requirements']]
        identified_req_incorrect = [req for req in requirements if req not in challenge['expected_requirements']]
        missing_req = [req for req in challenge['expected_requirements'] if req not in requirements]

        identified_bug_correct = [bug for bug in bugs if bug in challenge['expected_bugs']]
        identified_bug_incorrect = [bug for bug in bugs if bug not in challenge['expected_bugs']]
        missing_bugs = [bug for bug in challenge['expected_bugs'] if bug not in bugs]

        identified_imp_correct = []
        identified_imp_incorrect = []
        missing_imp = []

        if difficulty != "Easy":
            expected_improvements = challenge.get('expected_improvements', [])
            if improvements:
                identified_imp_correct = [imp for imp in improvements if imp in expected_improvements]
                identified_imp_incorrect = [imp for imp in improvements if imp not in expected_improvements]
            # Ensure we don't try to find missing improvements if no expected ones are defined
            missing_imp = [imp for imp in expected_improvements if imp not in (improvements if improvements else [])]
            # Handle the case where no improvements are expected but it's not Easy
            if not expected_improvements and improvements: # User provided improvements, but none were expected
                 identified_imp_incorrect.extend(improvements) # Treat all user-provided as incorrect
            elif not expected_improvements: # No expected improvements and user provided none
                pass # This is fine, imp_score_raw will be 0, correctly weighted out

        # Calculate scores as before
        req_score_raw = len(identified_req_correct) / len(challenge['expected_requirements']) if challenge['expected_requirements'] else 1
        bug_score_raw = len(identified_bug_correct) / len(challenge['expected_bugs']) if challenge['expected_bugs'] else 1

        imp_score_raw = 0
        if difficulty != "Easy":
            if challenge.get('expected_improvements'):
                imp_score_raw = len(identified_imp_correct) / len(challenge['expected_improvements'])
            else: # If no improvements are expected for Medium/Hard, and it's not Easy, give full score for that criteria.
                imp_score_raw = 1 

        total_weighted_score_proportion = (
            req_score_raw * criteria['correct_requirements'] +
            bug_score_raw * criteria['identified_bugs']
        )
        if difficulty != "Easy":
            total_weighted_score_proportion += imp_score_raw * criteria['code_improvements']

        scaled_score = round(total_weighted_score_proportion * max_score, 2)

        return {
            'score': scaled_score,
            'details': {
                'difficulty': difficulty,
                'max_score': max_score,
                'requirements': {
                    'expected': challenge['expected_requirements'],
                    'correct': identified_req_correct,
                    'incorrect_user_input': identified_req_incorrect,
                    'missing': missing_req
                },
                'bugs': {
                    'expected': challenge['expected_bugs'],
                    'correct': identified_bug_correct,
                    'incorrect_user_input': identified_bug_incorrect,
                    'missing': missing_bugs
                },
                'improvements': {
                    'expected': challenge.get('expected_improvements', []),
                    'correct': identified_imp_correct,
                    'incorrect_user_input': identified_imp_incorrect,
                    'missing': missing_imp
                } if difficulty != "Easy" else "N/A"
            }
        }

# --- Functions for Local JSON Data Storage ---
def save_submission_data(data: Dict):
    """Appends a new submission record to the JSON file."""
    if not os.path.exists(SUBMISSIONS_FILE):
        with open(SUBMISSIONS_FILE, 'w') as f:
            json.dump([], f)

    with open(SUBMISSIONS_FILE, 'r+') as f:
        file_data = json.load(f)
        file_data.append(data)
        f.seek(0)
        json.dump(file_data, f, indent=4)
        f.truncate()

def load_all_submission_data() -> List[Dict]:
    """Loads all submission records from the JSON file."""
    if not os.path.exists(SUBMISSIONS_FILE):
        return []
    with open(SUBMISSIONS_FILE, 'r') as f:
        return json.load(f)

def main():
    st.set_page_config(page_title="ALGO PROTOCOLS", layout="wide")

    # Initialize session state variables
    if 'challenge_system' not in st.session_state:
        st.session_state.challenge_system = LegacyJavaCodeChallenge()
        st.session_state.current_challenge = None
        st.session_state.show_hints = False
        st.session_state.submitted = False
        st.session_state.candidate_email = ""
        st.session_state.team_name = ""
        st.session_state.start_time = None
        st.session_state.time_up = False
        st.session_state.current_question = 0
        st.session_state.requirements = "" # Stores final submitted answers for a challenge
        st.session_state.bugs = ""         # Stores final submitted answers for a challenge
        st.session_state.improvements = "" # Stores final submitted answers for a challenge
        st.session_state.answers_saved = False # Flag for current form submission
        st.session_state.last_submission_details = None
        st.session_state.completed_challenges = {
            'Easy': {'completed': False, 'score': 0, 'answers': {}, 'evaluation_details': {}},
            'Medium': {'completed': False, 'score': 0, 'answers': {}, 'evaluation_details': {}},
            'Hard': {'completed': False, 'score': 0, 'answers': {}, 'evaluation_details': {}}
        }
        st.session_state.total_score = 0
        st.session_state.current_difficulty = "Easy"
        st.session_state.show_admin_dashboard = False

        # Initialize text area values for each question
        st.session_state.answer_requirements = ""
        st.session_state.answer_bugs = ""
        st.session_state.answer_improvements = ""


    challenge_system = st.session_state.challenge_system

    st.title("ALGO PROTOCOLS ")
    st.markdown("""
    ### Complete all 3 challenges (Easy, Medium, Hard) for maximum 100 points
    For each challenge:
    1. Identify requirements
    2. Find bugs
    3. Suggest improvements (for Medium/Hard)
    """)

    # Timer display for challenge duration
    if st.session_state.start_time and not st.session_state.time_up and not st.session_state.submitted:
        elapsed = datetime.now() - st.session_state.start_time
        remaining = timedelta(minutes=45) - elapsed
        if remaining.total_seconds() <= 0:
            st.session_state.time_up = True
            st.error("⏰ Time's up! Your answers are saved, please click 'Submit Final' if you haven't yet.")
        else:
            mins, secs = divmod(int(remaining.total_seconds()), 60)
            # Use st.empty to update the timer in place
            timer_placeholder = st.empty()
            timer_placeholder.warning(f"⏱️ Time left: {mins:02d}:{secs:02d}")
            # Rerun every second to update timer, but only if a challenge is active
            # This is handled implicitly by Streamlit's rerunning on input changes,
            # or explicitly by the form submission. We don't need a constant rerun() here.

    # Sidebar
    with st.sidebar:
        st.header("Team Info")
        st.session_state.team_name = st.text_input("Team Name (Required)", value=st.session_state.team_name)
        st.session_state.candidate_email = st.text_input("Your Email (Required)", value=st.session_state.candidate_email)

        st.header("Progress")
        for difficulty in ['Easy', 'Medium', 'Hard']:
            status = "✅" if st.session_state.completed_challenges[difficulty]['completed'] else "❌"
            score = st.session_state.completed_challenges[difficulty]['score']
            max_score = challenge_system.max_scores[difficulty]
            st.write(f"{status} {difficulty}: **{score}/{max_score}**") # Emphasize score here

        st.write(f"**Total Score:** {st.session_state.total_score}/100")

        st.header("Controls")
        if st.button("🎯 Start New Challenge Set"):
            if not st.session_state.team_name or not st.session_state.candidate_email:
                st.error("Please provide team name and email before starting.")
                return 

            # Reset all challenge-specific session state variables
            st.session_state.completed_challenges = {
                'Easy': {'completed': False, 'score': 0, 'answers': {}, 'evaluation_details': {}},
                'Medium': {'completed': False, 'score': 0, 'answers': {}, 'evaluation_details': {}},
                'Hard': {'completed': False, 'score': 0, 'answers': {}, 'evaluation_details': {}}
            }
            st.session_state.total_score = 0
            st.session_state.current_difficulty = "Easy"
            st.session_state.current_challenge = challenge_system.get_challenge_by_difficulty("Easy")
            st.session_state.start_time = datetime.now()
            st.session_state.time_up = False
            st.session_state.submitted = False
            st.session_state.current_question = 0
            st.session_state.requirements = ""
            st.session_state.bugs = ""
            st.session_state.improvements = ""
            st.session_state.answers_saved = False
            st.session_state.show_hints = False
            st.session_state.last_submission_details = None
            
            # Reset current question text areas as well
            st.session_state.answer_requirements = ""
            st.session_state.answer_bugs = ""
            st.session_state.answer_improvements = ""

            st.rerun()

        if st.session_state.current_challenge and st.button("💡 Toggle Hints"):
            st.session_state.show_hints = not st.session_state.show_hints
            st.rerun()

        # Admin controls
        if st.session_state.candidate_email == "arunsukumar03": # Admin email check
            st.markdown("---")
            st.header("Admin View")
            if st.button("📈 Show All Scores (Admin Only)"):
                st.session_state.show_admin_dashboard = True
                st.rerun()
            if st.session_state.get('show_admin_dashboard', False) and st.button("Hide Admin View"):
                st.session_state.show_admin_dashboard = False
                st.rerun()

    # Admin Dashboard
    if st.session_state.get('show_admin_dashboard', False) and st.session_state.candidate_email == "arunsukumar03":
        st.subheader("📊 All Challenge Submissions (Admin Dashboard)")
        all_submissions = load_all_submission_data()

        if not all_submissions:
            st.info("No submissions recorded yet.")
        else:
            # Flatten the submission data for DataFrame display
            flattened_data = []
            for sub in all_submissions:
                flat_sub = {
                    'submission_time': sub.get('submission_time'),
                    'team_name': sub.get('team_name'),
                    'candidate_email': sub.get('candidate_email'),
                    'total_score': sub.get('total_score'),
                    'easy_score': sub.get('easy_score', 0),
                    'medium_score': sub.get('medium_score', 0),
                    'hard_score': sub.get('hard_score', 0),
                    'requirements_submitted': sub.get('requirements', 'N/A'),
                    'bugs_submitted': sub.get('bugs', 'N/A'),
                    'improvements_submitted': sub.get('improvements', 'N/A')
                }
                flattened_data.append(flat_sub)

            df = pd.DataFrame(flattened_data)
            
            # Convert submission_time to datetime and format
            df['submission_time'] = pd.to_datetime(df['submission_time']).dt.strftime('%Y-%m-%d %H:%M:%S')

            st.dataframe(df.set_index('submission_time'), use_container_width=True)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download All Submissions as CSV",
                data=csv,
                file_name="all_challenge_submissions.csv",
                mime="text/csv",
            )
        st.stop() # Stop further execution if admin dashboard is active

    # Challenge Navigation
    if not st.session_state.current_challenge and not any(c['completed'] for c in st.session_state.completed_challenges.values()):
        st.info("👈 Enter team info and click '🎯 Start New Challenge Set' to begin!")
        st.stop()

    # Challenge Selection Buttons
    if st.session_state.current_challenge or any(c['completed'] for c in st.session_state.completed_challenges.values()):
        cols = st.columns(3)
        for i, difficulty in enumerate(['Easy', 'Medium', 'Hard']):
            with cols[i]:
                if st.session_state.completed_challenges[difficulty]['completed']:
                    # Display the score prominently after completion
                    st.success(f"✅ {difficulty} Completed")
                    st.markdown(f"**Score: {st.session_state.completed_challenges[difficulty]['score']}/{challenge_system.max_scores[difficulty]}**")
                # Logic to enable starting the next difficulty after previous is completed
                elif (difficulty == "Easy" or 
                      (difficulty == "Medium" and st.session_state.completed_challenges['Easy']['completed']) or
                      (difficulty == "Hard" and st.session_state.completed_challenges['Medium']['completed'])):
                    if st.button(f"Start {difficulty} Challenge"):
                        st.session_state.current_difficulty = difficulty
                        st.session_state.current_challenge = challenge_system.get_challenge_by_difficulty(difficulty)
                        st.session_state.current_question = 0
                        st.session_state.requirements = ""
                        st.session_state.bugs = ""
                        st.session_state.improvements = ""
                        st.session_state.answers_saved = False
                        st.session_state.show_hints = False
                        
                        # IMPORTANT: When starting a new challenge, reset the text area values
                        # to what was last saved for that challenge, or empty if new.
                        # This avoids carrying over answers from previous challenges.
                        st.session_state.answer_requirements = st.session_state.completed_challenges[difficulty]['answers'].get('requirements', "")
                        st.session_state.answer_bugs = st.session_state.completed_challenges[difficulty]['answers'].get('bugs', "")
                        st.session_state.answer_improvements = st.session_state.completed_challenges[difficulty]['answers'].get('improvements', "")

                        st.rerun()
                else:
                    st.warning(f"🔒 Complete {['Easy', 'Medium'][i-1]} first")

    # Current Challenge Display
    if st.session_state.current_challenge and not st.session_state.completed_challenges[st.session_state.current_difficulty]['completed']:
        challenge = st.session_state.current_challenge
        difficulty = st.session_state.current_difficulty
        max_score = challenge_system.max_scores[difficulty]

        st.subheader(f"🧩 {challenge['name']} ({difficulty} - {max_score} points)")

        with st.expander("📜 View Java Code", expanded=True):
            st.code(challenge['code'], language='java')

        if st.session_state.show_hints:
            st.subheader("❓ Hints")
            for hint in challenge['hints']:
                st.write(f"- {hint}")

        # Questions definition
        questions = [
            {"title": "1. Requirements (one per line)", "key": "requirements"},
            {"title": "2. Bugs (one per line)", "key": "bugs"}
        ]
        if difficulty != "Easy":
            questions.append({"title": "3. Improvements (one per line)", "key": "improvements"})

        # Ensure current_question index is within bounds
        if st.session_state.current_question >= len(questions):
            st.session_state.current_question = len(questions) - 1
        if st.session_state.current_question < 0:
            st.session_state.current_question = 0

        st.progress((st.session_state.current_question + 1) / len(questions))
        st.caption(f"Question {st.session_state.current_question + 1} of {len(questions)}")

        # Navigation buttons
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.session_state.current_question > 0:
                if st.button("⬅️ Previous", key="prev_button"):
                    # Save current answer before moving
                    current_q_key = questions[st.session_state.current_question]["key"]
                    st.session_state[f"answer_{current_q_key}"] = st.session_state[f"textarea_{f'answer_{current_q_key}'}"]
                    st.session_state.current_question -= 1
                    st.session_state.answers_saved = True # Assume saved when navigating back/forth
                    st.rerun()
        with col2:
            # Disable next if not saved for current question (only applies if not on last question)
            next_disabled = (not st.session_state.answers_saved) and (st.session_state.current_question < len(questions) - 1)
            if st.session_state.current_question < len(questions) - 1:
                if st.button("➡️ Next", disabled=next_disabled, key="next_button"):
                    # Save current answer before moving
                    current_q_key = questions[st.session_state.current_question]["key"]
                    st.session_state[f"answer_{current_q_key}"] = st.session_state[f"textarea_{f'answer_{current_q_key}'}"]
                    st.session_state.current_question += 1
                    st.session_state.answers_saved = False # Reset saved state for the new question
                    st.rerun()

        # The actual question form
        current_q = questions[st.session_state.current_question]
        
        # Use a consistent key for the form
        form_key = f"challenge_form_{difficulty}_{st.session_state.current_question}"

        with st.form(key=form_key):
            st.subheader(current_q["title"])

            answer_key_for_textarea = f"answer_{current_q['key']}"
            
            # Ensure the textarea's value is always initialized correctly based on session state
            # This is crucial for navigation: when you go "back" or "next", the textarea
            # should reflect the previously entered text for that specific question.
            current_answer_value = getattr(st.session_state, answer_key_for_textarea, "")

            answer = st.text_area(
                "Your answer",
                value=current_answer_value,
                height=150,
                key=f"textarea_{answer_key_for_textarea}" # Unique key for the text area itself
            )

            submit_label = "🚀 Submit Final" if st.session_state.current_question == len(questions) - 1 else "💾 Save Answer"
            submitted = st.form_submit_button(submit_label, disabled=st.session_state.time_up)

            if submitted:
                if not st.session_state.team_name or not st.session_state.candidate_email:
                    st.error("❗ Please provide team name and your email in the sidebar to proceed.")
                    return 
                
                # Save the answer from the current text area into its designated session state variable
                st.session_state[answer_key_for_textarea] = answer.strip() # Strip whitespace

                # Also update the final 'requirements', 'bugs', 'improvements' for scoring
                # These store the cumulative answers for the entire challenge
                setattr(st.session_state, current_q["key"], answer.strip())
                
                st.session_state.answers_saved = True # Mark current question's answer as saved

                if st.session_state.current_question == len(questions) - 1:
                    # This is the final submit for the current challenge
                    
                    # Ensure all answers for this challenge are captured before evaluation
                    # This is important if user jumps to last question and submits
                    st.session_state.requirements = st.session_state.answer_requirements.strip()
                    st.session_state.bugs = st.session_state.answer_bugs.strip()
                    st.session_state.improvements = st.session_state.answer_improvements.strip()

                    evaluation_result = challenge_system.evaluate_response(
                        challenge['id'],
                        [r.strip() for r in st.session_state.requirements.split('\n') if r.strip()],
                        [b.strip() for b in st.session_state.bugs.split('\n') if b.strip()],
                        [i.strip() for i in st.session_state.improvements.split('\n') if i.strip()] if difficulty != "Easy" else None
                    )

                    score = evaluation_result['score']
                    evaluation_details = evaluation_result['details']

                    st.session_state.completed_challenges[difficulty] = {
                        'completed': True,
                        'score': score,
                        'answers': {
                            'requirements': st.session_state.requirements,
                            'bugs': st.session_state.bugs,
                            'improvements': st.session_state.improvements if difficulty != "Easy" else "N/A"
                        },
                        'evaluation_details': evaluation_details
                    }

                    st.session_state.total_score = sum(
                        c['score'] for c in st.session_state.completed_challenges.values()
                    )
                    
                    # Prepare submission data for saving (cumulative for all challenges)
                    submission_data = {
                        "submission_time": datetime.now().isoformat(),
                        "team_name": st.session_state.team_name,
                        "candidate_email": st.session_state.candidate_email,
                        "total_score": st.session_state.total_score,
                        "easy_score": st.session_state.completed_challenges['Easy']['score'],
                        "medium_score": st.session_state.completed_challenges['Medium']['score'] if st.session_state.completed_challenges['Medium']['completed'] else 0,
                        "hard_score": st.session_state.completed_challenges['Hard']['score'] if st.session_state.completed_challenges['Hard']['completed'] else 0,
                        # Store all answers for submission file, even if N/A for Easy improvements
                        "requirements": (st.session_state.completed_challenges['Easy']['answers'].get('requirements', "") + "\n" +
                                        st.session_state.completed_challenges['Medium']['answers'].get('requirements', "") + "\n" +
                                        st.session_state.completed_challenges['Hard']['answers'].get('requirements', "")).strip(),
                        "bugs": (st.session_state.completed_challenges['Easy']['answers'].get('bugs', "") + "\n" +
                                st.session_state.completed_challenges['Medium']['answers'].get('bugs', "") + "\n" +
                                st.session_state.completed_challenges['Hard']['answers'].get('bugs', "")).strip(),
                        "improvements": (st.session_state.completed_challenges['Medium']['answers'].get('improvements', "") + "\n" +
                                        st.session_state.completed_challenges['Hard']['answers'].get('improvements', "")).strip()
                   
