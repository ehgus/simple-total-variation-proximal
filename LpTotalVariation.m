classdef LpTotalVariation < OptRegularizer
    properties
        weight
        p
        niter
        norm_weight
        % derived parameter
        norm
        x_tmp
        z_tmp
        z
    end
    methods
        function obj=LpTotalVariation(weight, p, niter, norm_weight)
            obj.weight=weight;
            obj.p=p;
            obj.niter=niter;
            obj.norm_weight=norm_weight;
            obj.norm=LpUnitBall(round(1/(1-1/p))); % projection of unit ball
        end
        function create_tmp_arrays(obj, x)
            if isempty(obj.x_tmp) || any(size(obj.x_tmp) ~= size(x))
                x_shape = size(x);
                z_shape = horzcat(length(x_shape), x_shape);
                obj.x_tmp = zeros(x_shape, 'like', x);
                obj.z = zeros(z_shape, 'like', x);
                obj.z_tmp = zeros(z_shape, 'like', x);
            end
        end
        function y=proximal(obj, y, x)
            w = obj.weight;
            v = obj.norm_weight;
            % y = x +  w(∇^T)z
            % where z = argmin[v|-z|_p* + v*(w/2|(∇^T)z|^2_2+(z^T)∇x)]
            % See Benchettou, Oumaima, Abdeslem Hafid Bentbib, and Abderrahman Bouhamidi. "An accelerated tensorial double proximal gradient method for total variation regularization problem." Journal of Optimization Theory and Applications 198.1 (2023): 111-134.
            % validation
            assert(strcmp(class(y), class(y)), "Input arguments should be the same class")
            if isgpuarray(x)
                spatial_diff = @spatial_diff_cuda;
                spatial_diff_T = @spatial_diff_T_cuda;
            else
                spatial_diff = @spatial_diff_cpu;
                spatial_diff_T = @spatial_diff_T_cpu;
            end
            % calculate z
            % step 1: initialize empty z
            % if x is (X,Y,Z,) shape, z0 is (X,Y,Z,3) where fourth dimension if for saving each partial differentiation
            obj.create_tmp_arrays(x);
            % step 2: iterative find z
            % where z = argmin(1/2|(∇^T)p|^2_2+(p^T)∇x + |-z|_p*)
            if isgpuarray(x)
                obj.z = lp_total_variation_cuda(x, w, v, obj.niter, obj.norm.p);
            else
                for idx=1:obj.niter
                    % Evaluate diff value of v*(w/2|(∇^T)z|^2_2+(z^T)∇x) -> diff = v∇(w(∇^T)z+x)
                    obj.x_tmp = x + w .* spatial_diff_T(obj.x_tmp, obj.z);
                    obj.z_tmp = v .* spatial_diff(obj.z_tmp, obj.x_tmp);
                    % Apply diff value
                    obj.z_tmp = obj.z - obj.z_tmp;
                    % Apply proximal
                    obj.z = v*obj.norm.projection(obj.z, obj.z_tmp ./ v);
                end
            end
            % step 3: calculate y
            % y = x +  w(∇^T)z
            y = spatial_diff_T(y, obj.z);
            y = x + w .* y;
        end
    end
end