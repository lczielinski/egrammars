#lang s-exp herbie/syntax/platform-language

;;; Arithmetic-only Herbie platform matching the egrammars subset:
;;; + - * / sqrt neg, plus comparisons/if for regimes. No fma, no libm specials.
;;; Costs copied from Herbie's math platform.

(define move-cost    0.02333600000000001)
(define fl-move-cost (* move-cost 4))

(define-representation <bool> #:cost move-cost)

(define-operations () <bool>
  [TRUE  #:spec (TRUE)  #:impl (const true)  #:fpcore TRUE  #:cost move-cost]
  [FALSE #:spec (FALSE) #:impl (const false) #:fpcore FALSE #:cost move-cost])

(define-operations ([x <bool>] [y <bool>]) <bool>
  [and #:spec (and x y) #:impl (lambda v (andmap values v)) #:cost move-cost]
  [or  #:spec (or x y)  #:impl (lambda v (ormap values v))  #:cost move-cost])

(define-operation (not [x <bool>]) <bool>
  #:spec (not x) #:impl not #:cost move-cost)

(define-representation <binary64> #:cost fl-move-cost)

(define-operation (if.f64 [c <bool>] [t <binary64>] [f <binary64>]) <binary64>
  #:spec (if c t f) #:impl if-impl
  #:cost (if-cost move-cost))

(define-operations ([x <binary64>] [y <binary64>]) <bool>
  [==.f64 #:spec (== x y) #:impl =          #:cost fl-move-cost]
  [!=.f64 #:spec (!= x y) #:impl (negate =) #:cost fl-move-cost]
  [<.f64  #:spec (< x y)  #:impl <          #:cost fl-move-cost]
  [>.f64  #:spec (> x y)  #:impl >          #:cost fl-move-cost]
  [<=.f64 #:spec (<= x y) #:impl <=         #:cost fl-move-cost]
  [>=.f64 #:spec (>= x y) #:impl >=         #:cost fl-move-cost])

(define-operations () <binary64> #:fpcore (! :precision binary64 _)
  [PI.f64   #:spec (PI)       #:impl (const pi)      #:fpcore PI       #:cost fl-move-cost]
  [E.f64    #:spec (E)        #:impl (const (exp 1)) #:fpcore E        #:cost fl-move-cost]
  [INFINITY #:spec (INFINITY) #:impl (const +inf.0)  #:fpcore INFINITY #:cost fl-move-cost]
  [NAN.f64  #:spec (NAN)      #:impl (const +nan.0)  #:fpcore NAN      #:cost fl-move-cost])

(define-operation (neg.f64 [x <binary64>]) <binary64>
  #:spec (neg x) #:impl -
  #:fpcore (! :precision binary64 (- x)) #:cost 0.096592)

(define-operations ([x <binary64>] [y <binary64>]) <binary64> #:fpcore (! :precision binary64 _)
  [+.f64 #:spec (+ x y) #:impl + #:cost 0.164604]
  [-.f64 #:spec (- x y) #:impl - #:cost 0.15163999999999997]
  [*.f64 #:spec (* x y) #:impl * #:cost 0.20874800000000002]
  [/.f64 #:spec (/ x y) #:impl / #:cost 0.26615199999999994])

(define-operations ([x <binary64>]) <binary64> #:fpcore (! :precision binary64 _)
  [sqrt.f64 #:spec (sqrt x) #:impl (from-libm 'sqrt) #:cost 0.191872])
