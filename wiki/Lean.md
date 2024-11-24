# LEAN proof assistant

- [Proof General](https://proofgeneral.github.io/) — Emacs package for running proof assistants.
- [Formalizing 100 Theorems](https://www.cs.ru.nl/~freek/100/) | cs.ru.nl — see how the formalization is going.
- [The Isabelle2024 Library](https://isabelle.in.tum.de/dist/library/) | isabelle.in.tum.de — see Isabelles' sources for a lot of theorems! It's fully interactive, clickable, and generally a fun browsing.
- [10 minute Lean tutorial : proving logical propositions](https://www.youtube.com/watch?v=POHVMMG7pqE&list=PL88g1zsvCrjexLVWaHTnXs23kuwDUZIbL) | youtube.com

Lean has a [category theory library](https://leanprover-community.github.io/mathlib4_docs/Mathlib/CategoryTheory/Action.html)!

### Basic Syntax

#### Comments
```lean
-- This is a single-line comment
/- This is a
   multi-line comment -/
```

#### Variables and Constants
```lean
variable (x : ℕ) -- Declare a variable x of type ℕ (natural number)
constant p : Prop -- Declare a constant p of type Prop (proposition)
```

#### Functions
```lean
def add (x y : ℕ) : ℕ := x + y -- Define a function add that takes two natural numbers and returns their sum
```

#### Types
```lean
#check ℕ -- Check the type of ℕ (natural numbers)
#check ℤ -- Check the type of ℤ (integers)
#check ℚ -- Check the type of ℚ (rational numbers)
#check ℝ -- Check the type of ℝ (real numbers)
```

### Propositions and Proofs

#### Propositions
```lean
variable (P Q : Prop) -- Declare two propositions P and Q
```

#### Logical Connectives
```lean
#check P ∧ Q -- Conjunction (P and Q)
#check P ∨ Q -- Disjunction (P or Q)
#check P → Q -- Implication (if P then Q)
#check ¬P -- Negation (not P)
#check P ↔ Q -- Equivalence (P if and only if Q)
```

#### Proofs
```lean
example : P → P :=
  assume h : P,
  h

example : P ∧ Q → P :=
  assume h : P ∧ Q,
  h.left

example : P ∨ Q → (P → R) → (Q → R) → R :=
  assume h : P ∨ Q,
  assume hP : P → R,
  assume hQ : Q → R,
  or.elim h hP hQ
```

### Tactics

#### Basic Tactics
```lean
example : P → P :=
begin
  intro h, -- Introduce the hypothesis h : P
  exact h, -- Prove the goal using h
end

example : P ∧ Q → P :=
begin
  intro h, -- Introduce the hypothesis h : P ∧ Q
  cases h with hP hQ, -- Split h into hP : P and hQ : Q
  exact hP, -- Prove the goal using hP
end

example : P ∨ Q → (P → R) → (Q → R) → R :=
begin
  intros h hP hQ, -- Introduce hypotheses h : P ∨ Q, hP : P → R, hQ : Q → R
  cases h with hP' hQ', -- Split h into hP' : P or hQ' : Q
  { exact hP hP' }, -- Prove the goal using hP and hP'
  { exact hQ hQ' }, -- Prove the goal using hQ and hQ'
end
```

#### Common Tactics
```lean
exact h -- Prove the goal using hypothesis h
intro h -- Introduce a hypothesis h
apply h -- Apply hypothesis h to the goal
split -- Split a conjunction (∧) or equivalence (↔) into two goals
left -- Prove a disjunction (∨) by proving the left side
right -- Prove a disjunction (∨) by proving the right side
cases h -- Split a hypothesis h into its components (e.g., ∧, ∨, ∃)
assumption -- Prove the goal using an existing hypothesis
contradiction -- Prove the goal by deriving a contradiction
refl -- Prove equality by reflexivity
rw h -- Rewrite the goal using equality h
simp -- Simplify the goal using known lemmas
```

### Inductive Types

#### Defining Inductive Types
```lean
inductive my_type : Type
| constructor1 : my_type
| constructor2 : ℕ → my_type

#check my_type.constructor1 -- Check the type of constructor1
#check my_type.constructor2 5 -- Check the type of constructor2 applied to 5
```

#### Pattern Matching
```lean
def my_function : my_type → ℕ
| my_type.constructor1 := 0
| (my_type.constructor2 n) := n + 1
```

### Structures and Classes

#### Defining Structures
```lean
structure point (α : Type) :=
(x : α) (y : α)

#check point ℕ -- Check the type of point applied to ℕ
```

#### Defining Classes
```lean
class has_add (α : Type) :=
(add : α → α → α)

instance : has_add ℕ :=
{ add := nat.add }
```

### Meta-Programming

#### Attributes
```lean
@[simp] lemma my_lemma : P ↔ Q := ... -- Mark a lemma for use in the simplifier
```

#### Macros
```lean
macro "my_macro" : tactic => `(tactic| exact 0)

example : ℕ :=
begin
  my_macro,
end
```

### Interactive Mode

#### Commands
```lean
#print my_theorem -- Print the definition of my_theorem
#eval my_function 5 -- Evaluate my_function applied to 5
#check my_type -- Check the type of my_type
```

#### Debugging
```lean
set_option pp.all true -- Enable detailed printing for debugging
```

This cheatsheet covers the basics of Lean syntax and some common constructs. For more advanced topics, refer to the official Lean documentation and tutorials.