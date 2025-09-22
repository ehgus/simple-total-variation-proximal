classdef LpUnitBall < OptProjection
    properties
        p
    end
    methods
        function obj=LpUnitBall(p)
            obj.p=p;
        end
        function y=projection(obj, y, x)
            % y, x: (X1, X2, ..., Xp, p) shape.
            % For each points (x1, x2, ..., xp), each vector with length p is projection

            if obj.p == 0
                % L0 unit ball projection
                y = obj.proj_l0_ball(y, x);
            elseif obj.p == 1
                % L1 unit ball projection
                y = obj.proj_l1_ball(y, x);
            elseif obj.p == 2
                % L2 unit ball projection
                y = obj.proj_l2_ball(y, x);
            elseif isinf(obj.p)
                % L-infinity unit ball projection
                y = obj.proj_linf_ball(y, x);
            else
                error("p value other than 0, 1, 2, and Inf is not supported")
            end
        end

        function y = proj_l0_ball(~, y, x)
            % L0 ball projection
            [~, max_I] = max(x, [], ndims(x), "ComparisonMethod", "abs");
            y(:) = 0;
            y(max_I) = x(max_I);
            y = max(min(y, 1), -1);
        end

        function y = proj_l1_ball(~, ~, x)
            % L1 ball projection
            norm_x = mean(abs(x), ndims(x));
            proj_degree = max(norm_x - 1, 0);
            y = x - proj_degree .* sign(x) .* norm_x;
        end

        function y = proj_l2_ball(~, ~, x)
            norm_x = sqrt(sum(x.^2, 1));
            inv_scale = max(norm_x, 1);
            y = x ./ inv_scale;
        end

        function y = proj_linf_ball(~, ~, x)
            y = max(min(x, 1), -1);
        end
    end
end