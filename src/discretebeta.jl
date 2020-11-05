# I can't differentiate through the quantile function of the Beta distribution
# The partial derivatives of the inverse incomplete beta function with respect
# to α and β are not defined in diffrules, and we can't do D through the
# functions in `SpecialFunctions`.
using Base.Math: @horner
using SpecialFunctions

function _auxgam(x)
    if x < 0
        return -(1.0+(1+x)*(1+x)*_auxgam(1+x))/(1.0-x)
    else
        t = 2*x - 1.0
        return _chepolsum(t, SpecialFunctions.auxgam_coef)
    end
end

function _chepolsum(x, a::Array{T,1}) where T
    n = length(a)
    if n == 1
        return a[1]/2.0
    elseif n == 2
        return a[1]/2.0 + a[2]*x
    else
        tx = 2*x
        r = a[n]
        h = a[n-1] + r*tx
        for k = n-2:-1:2
            s=r
            r=h
            h=a[k]+r*tx-s
        end
        return a[1]/2.0 - r + h*x
    end
end


"""
    loggamma1p(a)

Compute ``log(\\Gamma(1+a))`` for `-1 < a <= 1`.
"""
function _loggamma1p(x::Float64)
    return -log1p(x*(x-1.0)*_auxgam(x))
end


function _beta_inc(a, b, x, epps=1e-15)
    ans = 0.0
    if x == 0.0
        return 0.0
    end
    a0 = min(a,b)
    b0 = max(a,b)
    if a0 >= 1.0
        z = a*log(x) - logbeta(a,b)
        ans = exp(z)/a
    else

        if b0 >= 8.0
            u = _loggamma1p(a0) + loggammadiv(a0,b0)
            z = a*log(x) - u
            ans = (a0/a)*exp(z)
            if ans == 0.0 || a <= 0.1*epps
                return ans
            end
        elseif b0 > 1.0
            u = _loggamma1p(a0)
            m = b0 - 1.0
            if m >= 1.0
                c = 1.0
                for i = 1:m
                    b0 -= 1.0
                    c *= (b0/(a0+b0))
                end
                u += log(c)
            end
            z = a*log(x) - u
            b0 -= 1.0
            apb = a0 + b0
            if apb > 1.0
                u = a0 + b0 - 1.0
                t = (1.0 + _rgamma1pm1(u))/apb
            else
                t = 1.0 + _rgamma1pm1(apb)
            end
            ans = exp(z)*(a0/a)*(1.0 + _rgamma1pm1(b0))/t
            if ans == 0.0 || a <= 0.1*epps
                return ans
            end
        else
        #PROCEDURE FOR A0 < 1 && B0 < 1
            ans = x^a
            if ans == 0.0
                return ans
            end
            apb = a + b
            if apb > 1.0
                u = a + b - 1.0
                z = (1.0 + _rgamma1pm1(u))/apb
            else
                z = 1.0 + _rgamma1pm1(apb)
            end
            c = (1.0 + _rgamma1pm1(a))*(1.0 + _rgamma1pm1(b))/z
            ans *= c*(b/apb)
            #label l70 start
            if ans == 0.0 || a <= 0.1*epps
                return ans
            end
        end
    end
    if ans == 0.0 || a <= 0.1*epps
        return ans
    end
    # COMPUTE THE SERIES

    sm = 0.0
    n = 0.0
    c = 1.0
    tol = epps/a
    n += 1.0
    c *= x*(1.0 - b/n)
    w = c/(a + n)
    sm += w
    while abs(w) > tol
        n += 1.0
        c *= x*(1.0 - b/n)
        w = c/(a+n)
        sm += w
    end
    return ans*(1.0 + a*sm)
end

function _rgamma1pm1(a)
    t=a
    rangereduce = a > 0.5
    t = rangereduce ? a-1 : a #-0.5<= t <= 0.5
    if t == 0.0
        return 0.0
    elseif t < 0.0
        top = @horner(t , -.422784335098468E+00 , -.771330383816272E+00 , -.244757765222226E+00 , .118378989872749E+00 , .930357293360349E-03 , -.118290993445146E-01 , .223047661158249E-02 , .266505979058923E-03 , -.132674909766242E-03)
        bot = @horner(t , 1.0 , .273076135303957E+00 , .559398236957378E-01)
        w = top/bot
        return rangereduce ? t*w/a : a*(w+1)
    else
        top = @horner(t , .577215664901533E+00 , -.409078193005776E+00 , -.230975380857675E+00 , .597275330452234E-01 , .766968181649490E-02 , -.514889771323592E-02 , .589597428611429E-03)
        bot = @horner(t , 1.0 , .427569613095214E+00 , .158451672430138E+00 , .261132021441447E-01 , .423244297896961E-02)
        w = top/bot
        return rangereduce ? (t/a)*(w - 1.0) : a*w
    end
end

"""
    beta_inc_inv(a,b,p,q,lb=logbeta(a,b)[1])

Computes inverse of incomplete beta function. Given `a`,`b` and ``I_{x}(a,b) =
p`` find `x` and return tuple `(x,y)`.  See also: [`beta_inc(a,b,x)`](@ref
SpecialFunctions.beta_inc)
"""
function _beta_inc_inv(a, b, p, q; lb = logbeta(a,b)[1])
    fpu = 1e-30
    x = p
    if p == 0.0
        return (0.0, 1.0)
    elseif p == 1.0
        return (1.0, 0.0)
    end

    #change tail if necessary

    if p > 0.5
        pp = q
        aa = b
        bb = a
        indx = true
    else
        pp = p
        aa = a
        bb = b
        indx = false
    end

    #Initial approx

    r = sqrt(-log(pp^2))
    pp_approx = r - @horner(r, 2.30753e+00, 0.27061e+00) / @horner(r, 1.0, .99229e+00, .04481e+00)

    if a > 1.0 && b > 1.0
        r = (pp_approx^2 - 3.0)/6.0
        s = 1.0/(2*aa - 1.0)
        t = 1.0/(2*bb - 1.0)
        h = 2.0/(s+t)
        w = pp_approx*sqrt(h+r)/h - (t-s)*(r + 5.0/6.0 - 2.0/(3.0*h))
        x = aa/ (aa+bb*exp(w^2))
    else
        r = 2.0*bb
        t = 1.0/(9.0*bb)
        t = r*(1.0-t+pp_approx*sqrt(t))^3
        if t <= 0.0
            x = -expm1((log((1.0-pp)*bb)+lb)/bb)
        else
            t = (4.0*aa+r-2.0)/t
            if t <= 1.0
                x = exp((log(pp*aa)+lb)/aa)
            else
                x = 1.0 - 2.0/(t+1.0)
            end
        end
    end

    #solve x using modified newton-raphson iteration

    r = 1.0 - aa
    t = 1.0 - bb
    pp_approx_prev = 0.0
    sq = 1.0
    prev = 1.0

    if x < 0.0001
        x = 0.0001
    end
    if x > .9999
        x = .9999
    end

    iex = max(-5.0/aa^2 - 1.0/pp^0.2 - 13.0, -30.0)
    acu = 10.0^iex

    #iterate
    while true
        pp_approx = _beta_inc(aa,bb,x)[1]
        xin = x
        pp_approx = (pp_approx-pp)*exp(lb+r*log(xin)+t*log1p(-xin))
        if pp_approx * pp_approx_prev <= 0.0
            prev = max(sq, fpu)
        end
        g = 1.0

        tx = x - g*pp_approx
        while true
            adj = g*pp_approx
            sq = adj^2
            tx = x - adj
            if (prev > sq && tx >= 0.0 && tx <= 1.0)
                break
            end
            g /= 3.0
        end

        #check if current estimate is acceptable

        if prev <= acu || pp_approx^2 <= acu
            x = tx
            return indx ? (1.0 - x, x) : (x, 1.0-x)
        end

        if tx == x
            return indx ? (1.0 - x, x) : (x, 1.0-x)
        end

        x = tx
        pp_approx_prev = pp_approx
    end
end

_beta_inc_inv(a, b, p) = _beta_inc_inv(a, b, p, 1.0-p)

qbeta(α::Real, β::Real, q::Real) = _beta_inc_inv(α, β, q, 1. -q)[1]

function discretize_beta(α, β, K)
    qstart = 1.0/2K
    qend = 1. - 1.0/2K
    xs = [qbeta(α, β, q) for q in qstart:(1/K):qend]
    xs *= (α/(α+β))*K/sum(xs)
end
