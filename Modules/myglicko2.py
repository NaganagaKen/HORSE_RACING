"""
Copyright (c) 2009 Ryan Kirkman

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
"""

"""
myglicko2.py
pipでインストールしたglicko2の
修正版Glicko2を実装した。

- _E関数の計算方法を修正
- _f関数の計算方法を修正
"""


import math

class Player:
    # Class attribute
    # The system constant, which constrains
    # the change in volatility over time.
    _tau = 0.3
    _EPS = 1e-4

    def getRating(self):
        return (self.__rating * 173.7178) + 1500 

    def setRating(self, rating):
        self.__rating = (rating - 1500) / 173.7178

    rating = property(getRating, setRating)

    def getRd(self):
        return self.__rd * 173.7178

    def setRd(self, rd):
        self.__rd = rd / 173.7178

    rd = property(getRd, setRd)
     
    def __init__(self, rating = 1500, rd = 350, vol = 0.06):
        # For testing purposes, preload the values
        # assigned to an unrated player.
        self.setRating(rating)
        self.setRd(rd)
        self.vol = vol
            
    def _preRatingRD(self):
        """
        Calculates and updates the player's rating deviation (RD) at the
        beginning of a rating period, then clamps it into a reasonable range.
        """
        self.__rd = math.sqrt(self.__rd ** 2 + self.vol ** 2)

        rd_floor = 40  / 173.7178   # ≒ 0.23  (external 40)
        rd_cap   = 350 / 173.7178   # ≒ 2.02  (external 350)
        self.__rd = max(min(self.__rd, rd_cap), rd_floor)

        
    def update_player(self, rating_list, RD_list, outcome_list):
        rating_list = [(r - 1500) / 173.7178 for r in rating_list]
        RD_list     = [rd / 173.7178 for rd in RD_list]

        self._preRatingRD()

        v     = self._v(rating_list, RD_list)
        delta = self._delta(rating_list, RD_list, outcome_list, v)
        self.vol = self._newVol(rating_list, RD_list, outcome_list, v)

        self.__rd = 1.0 / math.sqrt((1.0 / self.__rd ** 2) + (1.0 / v))
        rd_floor = 40  / 173.7178
        rd_cap   = 350 / 173.7178
        self.__rd = max(min(self.__rd, rd_cap), rd_floor)

        sum_term = sum(
            self._g(RD_list[i]) * (outcome_list[i] - self._E(rating_list[i], RD_list[i]))
            for i in range(len(rating_list))
        )
        self.__rating += self.__rd ** 2 * sum_term

        # external scale clamp
        ext_rating = min(self.getRating(), 3000.0)
        self.setRating(ext_rating)


    #step 5
    def _newVol(self, rating_list, RD_list, outcome_list, v):
        """
        Newton iteration (2012 revision) to obtain the new volatility σ.
        Uses class constant _EPS for convergence and clamps the result.
        """
        a  = math.log(self.vol ** 2)
        eps = self._EPS
        A = a

        delta = self._delta(rating_list, RD_list, outcome_list, v)
        tau   = self._tau
        if delta ** 2 > (self.__rd ** 2 + v):
            B = math.log(delta ** 2 - self.__rd ** 2 - v)
        else:
            k = 1
            while self._f(a - k * tau, delta, v, a) < 0.0:
                k += 1
            B = a - k * tau

        fA = self._f(A, delta, v, a)
        fB = self._f(B, delta, v, a)
        while abs(B - A) > eps:
            C  = A + (A - B) * fA / (fB - fA)
            fC = self._f(C, delta, v, a)
            if fC * fB <= 0:
                A, fA = B, fB
            else:
                fA /= 2.0
            B, fB = C, fC

        new_vol = math.exp(A / 2.0)
        return max(min(new_vol, 1.5), 1e-3)   # σ ∈ [1e-3, 1.5]
        
    def _f(self, x, delta, v, a):
        ex   = math.exp(x)
        phi2 = self._Player__rd ** 2          # ← RD²
        num  = ex * (delta**2 - phi2 - v - ex)
        den  = 2.0 * (phi2 + v + ex) ** 2
        return num / den - (x - a) / (self._tau ** 2)
        
    def _delta(self, rating_list, RD_list, outcome_list, v):
        """ The delta function of the Glicko2 system.
        
        _delta(list, list, list) -> float
        
        """
        tempSum = 0
        for i in range(len(rating_list)):
            tempSum += self._g(RD_list[i]) * (outcome_list[i] - self._E(rating_list[i], RD_list[i]))
        return v * tempSum
        
    def _v(self, rating_list, RD_list):
        """
        The v function of the Glicko-2 system with numerical guard.
        """
        temp_sum = 0.0
        for r, rd in zip(rating_list, RD_list):
            g = self._g(rd)
            E = self._E(r, rd)
            temp_sum += (g * g) * max(E * (1.0 - E), self._EPS)
        return 1.0 / max(temp_sum, self._EPS)
        
    def _E(self, p2rating, p2RD):
        g      = self._g(p2RD)
        expo   = -g * (self._Player__rating - p2rating)   # __rating は name-mangling で隠蔽されている
        if expo > 35:     # exp(-35) ≃ 6.3e-16
            return self._EPS
        if expo < -35:    # exp(35)  は安全
            return 1.0 - self._EPS
        return 1.0 / (1.0 + math.exp(expo))
        
    def _g(self, RD):
        """ The Glicko2 g(RD) function.
        
        _g() -> float
        
        """
        return 1 / math.sqrt(1 + 3 * math.pow(RD, 2) / math.pow(math.pi, 2))
        
    def did_not_compete(self):
        """ Applies Step 6 of the algorithm. Use this for
        players who did not compete in the rating period.

        did_not_compete() -> None
        
        """
        self._preRatingRD()
