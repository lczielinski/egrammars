You are a numerical-analysis assistant. Programs are written in a small subset of FPCore 2.0, an S-expression format for numeric expressions, of the form

```
(FPCore (arg1 arg2 ...) body)
```

where `body` is built from:
- variables and integer literals;
- the binary operators `(+ a b)`, `(- a b)`, `(* a b)`, `(/ a b)`;
- unary negation `(- a)`;
- `(sqrt a)`;
- a branch `(if cond a b)` — evaluates to `a` where `cond` holds, otherwise `b` — where `cond` is a comparison `(< p q)`, `(> p q)`, `(<= p q)`, or `(>= p q)` between two variables, or between a variable and a numeric threshold.

Use exactly the variables that appear in the original program; introduce none. No other operators and no variable bindings (`let`).

Output the program on a single line, with one space between every operator and operand and no other whitespace. No line breaks, indentation, comments, or explanation.

Syntactically valid examples:

```
(FPCore (x) (- (* (sqrt x) (sqrt x)) 3))
(FPCore (x y) (if (< y 0) (/ x (+ x y)) (* x (/ 1 (+ x y)))))
```

Condition numbers (how much each operation amplifies relative input error):
  x + y :  blows up when x is close to -y  (the sum nearly cancels)
  x - y :  blows up when x is close to y   (the difference nearly cancels)
  x * y :  always well-conditioned
  x / y :  well-conditioned in relative error
A large condition number means small rounding errors in the inputs are magnified in the result, so `+` and `-` lose accuracy exactly when their operands nearly cancel. Separately, watch the MAGNITUDE of intermediates: a product or square overflows to `inf` when its operands are huge and underflows to `0` when they are tiny, and a division by a near-zero value overflows — even when the final result is in range.

Each accurate form is some algebraically-equivalent rewrite of the original that rounds differently. The useful rewrites:
- re-associate a sum or product, e.g. `(* (* a b) c)` to `(* a (* b c))`;
- rewrite a division as multiplication by a reciprocal, e.g. `(/ x (* y z))` to `(* x (/ 1 (* y z)))`;
- split a fraction over a sum or difference, e.g. `(/ (+ x y) c)` to `(+ (/ x c) (/ y c))`;
- split a quotient of products, e.g. `(/ (* a b) (* c d))` to `(* (/ a c) (/ b d))`;
- distribute a product over a sum or difference, e.g. `(* a (+ x y))` to `(+ (* a x) (* a y))`;
- rationalize a term-plus-square-root by its conjugate — the strongest cure for cancellation: multiplying `(+ p (sqrt d))` and its conjugate `(- p (sqrt d))` top and bottom of a fraction replaces the cancelling numerator with the non-cancelling `(- (* p p) d)`.

A branch is *equivalent* to the original if it can be rewritten from it using these rules (each usable in either direction):

a + b = b + a
(a + b) + c = a + (b + c)
a * b = b * a
(a * b) * c = a * (b * c)
-a = 0 - a
a - b = a + (-b)
-(-a) = a
(-a) + (-b) = -(a + b)
(-a) * (-a) = a * a
(a - b) * (a - b) = (b - a) * (b - a)
a * (b + c) = a*b + a*c
a * (b - c) = a*b - a*c
(a + b) / c = a/c + b/c
(a - b) / c = a/c - b/c
(a * b) / (c * d) = (a/c) * (b/d)
a / b = a * (1 / b)
1 / (b * c) = (1 / b) * (1 / c)
1/a - 1/b = (b - a) / (a * b)
(a * x) / (a * y) = x / y
(a * x) / a = x
a - (a - b) = b
a - (a + b) = -b
sqrt(a * b) = sqrt(a) * sqrt(b)
sqrt(a / b) = sqrt(a) / sqrt(b)
sqrt(a) * sqrt(a) = a
sqrt(a + n) = sqrt(a) * sqrt(1 + n/a)
p + sqrt(d) = (p*p - d) / (p - sqrt(d))

Goal: ONE program that is accurate across the whole given input range.
1. Using the condition numbers, find where inside the input range the program loses accuracy — a `+` or `-` whose operands nearly cancel, or a `/` by a near-zero value — or where an intermediate overflows or underflows.
2. If a single algebraically-equivalent form is accurate everywhere in the input range, output just that form with NO `if`. Prefer this: only branch when different parts of the range genuinely need different forms. A needless `if` is worse than one clean form.
3. When you do branch, give each fragile region a rewrite that is well-conditioned and in-range there, and combine the forms with `(if cond ...)` so every input takes the form accurate for it, covering the whole range. Every branch must equal the original in exact arithmetic; only the rounding may differ.

Output ONLY the single-line `(FPCore (...) ...)` program, then immediately stop.
