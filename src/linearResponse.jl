# functions to be used in the context of the methods described in https://arxiv.org/abs/1907.10852
# still very much under development and undocumented

function linear_response_tensor(flowmap, u, p)
	DT = linearized_flow_autodiff(x -> flowmap(x,p), u)
	Tdot = u -> flowmap_parameter_derivative(flowmap, u, p)
	DTdot = linearized_flow_autodiff(Tdot, u)
	DTinv = inv(DT)
	return -Tensors.symmetric(DTinv ⋅ DTdot ⋅ Tensors.dott(DTinv))
end

linearized_flow_autodiff(flowmap, x) = Tensor{2,2}(ForwardDiff.jacobian(flowmap, x))

flowmap_parameter_derivative(flowmap, u, p) = ForwardDiff.jacobian(x -> flowmap(x[1:end - 1], x[end]), vcat(u, p))[:,end]


# adapted from adaptiveTOCollocationStiffnessMatrix
function adaptiveTOCollocationLinearResponseMatrix(ctx, flowmap, p; bdata::BoundaryData=BoundaryData())
    Tdot(u) = flowmap_parameter_derivative(flowmap, u, p)

    flow_map_images = [Vec{2}(flowmap(ctx.grid.nodes[i].x, p)) for i in 1:ctx.n]
    flow_map_t(j) = flow_map_images[j]

    new_ctx,new_bdata,_ = adaptiveTOFutureGrid(ctx,flow_map_t,bdata=bdata, flow_map_mode=1)

    # for now we assume volume preserving
    Tdots_at_dofs = [Vec{2}(Tdot(ctx.grid.nodes[i].x)) for i in bcdof_to_node(new_ctx,new_bdata)]

    L = assembleLinearResponseMatrix(new_ctx, Tdots_at_dofs, bdata=new_bdata)

    translation_table_new = bcdof_to_node(new_ctx,new_bdata)
    I, J, V = Main.CoherentStructures.findnz(L)
    I .= translation_table_new[I]
    J .= translation_table_new[J]
    L = Main.CoherentStructures.sparse(I,J,V,size(L,1),size(L,2))

    return 0.5*(L+L')
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

    index = 1 # quadrature point counter

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
                        dof_s = JFM.celldofs(cell)[s]
                        ∇φₛ = JFM.shape_gradient(cv, q, s)
                        Ke[k,l] += (∇φₖ ⋅ Tdots_at_dofs[dof_s])  * (∇φₛ ⋅ ∇φₗ) * dΩ
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