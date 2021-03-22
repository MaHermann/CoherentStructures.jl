# functions to be used in the context of the methods described in https://arxiv.org/abs/1907.10852
# still very much under development and undocumented

function linear_response_tensor_autodiff(flowmap, u, p)
	DT = linearized_transfer_function_autodiff(x -> flowmap(x,p), u)
	Tdot = u -> parameter_autodiff(flowmap, u, p)
	DTdot = linearized_transfer_function_autodiff(Tdot, u)
	DTinv = inv(DT)
	return -Tensors.symmetric(DTinv ⋅ DTdot ⋅ Tensors.dott(DTinv))
end

linearized_transfer_function_autodiff(flowmap, x) = Tensor{2,2}(ForwardDiff.jacobian(flowmap, x))

parameter_autodiff(flowmap, u, p) = ForwardDiff.jacobian(x -> flowmap(x[1:end - 1], x[end]), vcat(u, p))[:,end]

# adapted from adaptiveTOCollocationStiffnessMatrix
function adaptiveTOCollocationLinearResponseMatrix(ctx, flowmap, p; bdata::BoundaryData=BoundaryData())
	Tdot(u) = parameter_autodiff(transferfun, u, p)
	num_real_points = ctx.n
	flow_map_images = zeros(Vec{2}, num_real_points)
	for i in 1:num_real_points
		flow_map_images[i] = Vec{2}(flowmap(ctx.grid.nodes[i].x, p))
	end
	Tdots_at_dofs = [Vec{2}(Tdot(ctx.grid.nodes[j].x)) for j in bcdof_to_node(ctx, bdata)]
	flow_map_t(j) = flow_map_images[j]
    new_ctx,new_bdata,_ = adaptiveTOFutureGrid(ctx,flow_map_t,bdata=bdata, flow_map_mode=1)
	# for now we assume volume preserving
	return assembleLinearResponseMatrix(new_ctx, Tdots_at_dofs, bdata=new_bdata)
end

# adapted from _assembleStiffnesMatrix
function assembleLinearResponseMatrix(ctx, Tdots_at_dofs; bdata::BoundaryData=BoundaryData())
	cv = JFM.CellScalarValues(ctx.qr, ctx.ip, ctx.ip_geom)
	dh = ctx.dh
	K = JFM.create_sparsity_pattern(dh)
	a_K = JFM.start_assemble(K)
    dofs = zeros(Int, JFM.ndofs_per_cell(dh))
    n = JFM.getnbasefunctions(cv)
    Ke = zeros(n, n)

	index = 1

	@inbounds for cell in JFM.CellIterator(dh)
        fill!(Ke, 0)
        JFM.reinit!(cv, cell)
        for q in 1:JFM.getnquadpoints(cv)
            dΩ = JFM.getdetJdV(cv, q) * ctx.mass_weights[index]
            for k in 1:n
                ∇φₖ = JFM.shape_gradient(cv, q, k)
                for l in 1:n
                    ∇φₗ = JFM.shape_gradient(cv, q, l)
					for s in 1:n
					    # the node in dof order that corresponds to the s-th shape funciton of the cell
						dof_s = JFM.celldofs(cell)[s]
						∇φₛ = JFM.shape_gradient(cv, q, s)
						# this makes some very implicit assumptions about the order of celldofs that I really do not like
                   		Ke[k,l] += 0.5*(Tdots_at_dofs[dof_s] ⋅ ∇φₗ)  * (∇φₛ ⋅ ∇φₖ) * dΩ
						Ke[k,l] += 0.5*(Tdots_at_dofs[dof_s] ⋅ ∇φₖ)  * (∇φₛ ⋅ ∇φₗ) * dΩ
					end
                end
            end
            index += 1
        end
        JFM.celldofs!(dofs, cell)
        JFM.assemble!(a_K, dofs, Ke)
    end
	return applyBCS(ctx, K, bdata)
end
