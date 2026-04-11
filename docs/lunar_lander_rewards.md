# Lunar Lander (Gymnasium v2/v3) — פירוט תגמולים וצעדים לפרק

## מקור האמת

המימוש הרשמי הוא ב-[`gymnasium/envs/box2d/lunar_lander.py`](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/box2d/lunar_lander.py). בפרויקט זה משתמשים ב-`LunarLander-v3` (ראה `DEFAULT_ENV_ID` ב-`lunar_rl_common.py`). **נוסחת התגמול זהה ל-v2**; השינויים ב-v3 נוגעים בעיקר לתצפית, reset, ודטרמיניזם (לא לנוסחת ה-reward).

---

## 1. תגמול בכל צעד (per-step)

### א. Shaping (שינוי בפוטנציאל)

מחשבים פונקציית עזר `shaping` מהמצב הנורמלי (אחרי סימולציית הפיזיקה של הצעד):

```python
shaping = (
    -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
    - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
    - 100 * abs(state[4])
    + 10 * state[6]
    + 10 * state[7]
)
```

`state[0..7]` הוא וקטור התצפית (8 מימדים): מיקום x,y נורמלי, מהירויות נורמלות, זווית, מהירות זוויתית, ושני דגלים לרגל במגע עם הקרקע.

**התגמול מהשינוי ב-shaping** (מצעד שני ואילך; ב-`reset` מוגדר `prev_shaping = None`):

```python
if self.prev_shaping is not None:
    reward = shaping - self.prev_shaping
self.prev_shaping = shaping
```

| רכיב | תרומה ל-`shaping` (לפני הפרש) | משמעות |
|------|-------------------------------|--------|
| מרחק מהמשטח | `-100 * sqrt(x² + y²)` | קרוב יותר ללוח הנחיתה → ערך פוטנציאל גבוה יותר (פחות שלילי) |
| מהירות | `-100 * sqrt(vx² + vy²)` | איטי יותר → עדיף |
| זווית | `-100 * \|angle\|` | קרוב לאופק → עדיף |
| רגל על הקרקע | `+10` לכל רגל (`state[6]`, `state[7]` ∈ {0,1}) | עד שתי רגליים = עד `+20` בפוטנציאל |

**חשוב:** בפועל מקבלים את **הפרש** `Δshaping` בין צעד לצעד, לא את כל הסכום בכל פריים — תגמול מבוסס שינוי (potential-based shaping).

### ב. עלות מנועים ("דלק")

```python
reward -= m_power * 0.30
reward -= s_power * 0.03
```

| מנוע | מתי נדלק | `m_power` / `s_power` | עונש לצעד |
|------|-----------|----------------------|-----------|
| ראשי (main) | Discrete: פעולה `2`. Continuous: `action[0] > 0` | Discrete: `1.0`. Continuous: `(clip(a0,0,1)+1)*0.5` ∈ **[0.5, 1.0]** | **−0.30 × m_power** (ב-discrete: **−0.3** לפריים שבו המנוע דולק) |
| צדדי (side) | Discrete: `1` או `3`. Continuous: `\|action[1]\| > 0.5` | Discrete: `1.0`. Continuous: `clip(\|a1\|, 0.5, 1.0)` | **−0.03 × s_power** (ב-discrete: **−0.03** לפריים) |

הערה בקוד Gymnasium: נחיתה היוריסטית בערך **~−30** סה"כ מעונש מנוע ראשי לאורך פרק טיפוסי.

---

## 2. תגמול בסוף פרק (termination)

```python
if self.game_over or abs(state[0]) >= 1.0:
    terminated = True
    reward = -100
if not self.lander.awake:
    terminated = True
    reward = +100
```

| סיום | תנאי (בקצרה) | תגמול באותו צעד |
|------|----------------|------------------|
| התרסקות / יציאה מהמסך | גוף הנחתת פוגע בקרקע (`game_over`), או `\|x\|_norm ≥ 1` | **−100** |
| נחיתה מוצלחת | `not lander.awake` (מנוחה יציבה ב-Box2D) | **+100** |

אם שני התנאים מתקיימים באותו צעד, הסדר בקוד קובע: קודם **−100**, ואז אם `not awake` — **+100**.

---

## 3. פרויקט RL-LunarLander

ב-`lunar_rl_common.py` אין שכבת תגמול מותאמת אישית — רק עטיפות כמו `VecNormalize` (נרמול תצפית; `norm_reward` מוגדר בהתאם לשימוש). התגמול הבסיסי הוא של Gymnasium כפי למעלה.

---

## 4. כמה צעדים (אינטראקציות) לפרק בממוצע?

- **תקרה:** `gym.make("LunarLander-v3")` כולל בדרך כלל `TimeLimit` עם **`max_episode_steps = 1000`**.
- **ממוצע:** אין מספר קבוע מהסביבה — תלוי במדיניות. כדי ממוצע אמפירי יש למדוד `episode length` בהערכה על המודל הספציפי.

---

## 5. הערה ל-reward shaping עתידי

בבסיס כבר קיימים: פרש פוטנציאל (מיקום/מהירות/זווית/רגליים), עונשי דלק, ובונוס/קנס סופיים ±100. הרחבות מותאמות כדאי לתאם כדי למנוע כפילות ולשמור על סקיילינג סביר.
