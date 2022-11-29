# approx-model-or-approx-soln


## Pretty darn good control: when are approximate solutions better than approximate models?


Existing methods for optimal control methods struggle to deal with the
complexity commonly encountered in real-world systems, including
dimensionality, process error, model bias and heterogeneity.  Instead
of tackling these complexities directly, researchers have typically
sought to find exact optimal solutions to simplified models of the
processes in question. When is the optimal solution to a very
approximate, stylized model better than an approximate solution to a
more accurate model? This question has largely gone unanswered owing
to the difficulty of finding even approximate solutions in the case of
complex models.  Our approach draws on recent algorithmic and
computational advances in deep reinforcement learning. These methods
have hitherto focused on problems in games or robotic mechanics, which
operate under precisely known rules. We demonstrate the ability for
novel algorithms using deep neural networks to successfully
approximate such solutions (the "policy function" or control rule)
without knowing or ever attempting to infer a model for the process
itself. This powerful new technique lets us finally begin to answer
the question. We show that in many but not all cases, the optimal
policy for a carefully chosen over-simplified model can still
out-perform these novel algorithms trained to find approximate
solutions to simulations of a realistically complex system. Comparing
these two approaches can lead to insights in the importance of
real-world features of observation and process error, model biases,
and heterogeneity.
