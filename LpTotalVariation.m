classdef LpTotalVariation < OptRegularizer
    properties
        weight
        p
        niter
        norm_weight
        % derived parameter
        norm
    end
    methods
        function obj=LpTotalVariation(weight, p, niter, norm_weight)
            obj.weight=weight;
            obj.p=p;
            obj.niter=niter;
            obj.norm_weight=norm_weight;
            obj.norm=LpUnitBall(round(1/(1-1/p))); % projection of unit ball
        end
        function y=proximal(obj, y, x)
            w = obj.weight;
            v = obj.norm_weight;
            % y = x +  w(∇^T)z
            % where z = argmin[v|-z|_p* + v*(w/2|(∇^T)z|^2_2+(z^T)∇x)]
            % See Benchettou, Oumaima, Abdeslem Hafid Bentbib, and Abderrahman Bouhamidi. "An accelerated tensorial double proximal gradient method for total variation regularization problem." Journal of Optimization Theory and Applications 198.1 (2023): 111-134.
            
            % calculate z
            % step 1: initialize empty z
            % if x is (X,Y,Z,) shape, z0 is (X,Y,Z,3) where fourth dimension if for saving each partial differentiation
            x_shape = size(x);
            z_shape = horzcat(length(x_shape), x_shape);
            x_tmp = zeros(x_shape);
            z = zeros(z_shape);
            z_next = zeros(z_shape);
            diff = z_next;
            % step 2: iterative find z
            % where z = argmin(1/2|(∇^T)p|^2_2+(p^T)∇x + |-z|_p*)
            for idx=1:obj.niter
                % Evaluate diff value of v*(w/2|(∇^T)z|^2_2+(z^T)∇x) -> diff = v∇(w(∇^T)z+x)
                x_tmp = x + w * spatial_diff_T(x_tmp, z);
                diff = v*spatial_diff(diff, x_tmp);
                % Apply diff value
                z = z - diff;
                % Apply proximal
                z_next = v*obj.norm.projection(z_next, z/v);
                z(:) = z_next;
            end
            % step 3: calculate y
            % y = x +  w(∇^T)z
            y = spatial_diff_T(y, z);
            y = x + w .* y;
        end
    end
end