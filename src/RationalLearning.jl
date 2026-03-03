module RationalLearning

using Combinatorics
using Statistics
using Printf
using Symbolics
using DataFrames
using CSV
using LsqFit  # 新增：引入非线性最小二乘拟合包

export run_rational_learning

# ==============================================================================
# Helper Functions (保持原有修复了 axes 的逻辑)
# ==============================================================================

"""
Generate all power combinations for a single term where the sum of powers 
is not greater than `degree_value`, and the number of variables is `dim_value`.
"""
function get_all_single_term_power_combination(degree_value::Int, dim_value::Int)
    M = collect(Iterators.product(fill(0:degree_value, dim_value)...))
    M = filter(x -> sum(x) <= degree_value && sum(x) >= 1, M)
    MM = reduce(hcat, map(x -> collect(x), M))'
    return MM
end

"""
Generate an array of tuples representing the number of terms in numerator and denominator.
e.g., (1,0), (2,0), (1,1), etc.
"""
function get_the_array(num_numerator::Int, num_denominator::Int)
    array_num_terms = []
    for j in 0:num_denominator
        for i in 0:num_numerator
            push!(array_num_terms, (i, j))
        end
    end
    filter!(x -> x != (0,0), array_num_terms)
    return array_num_terms
end

"""
Obtain all power combinations and their corresponding split numbers.
"""
function get_power_combination(MM, x_max_term::Int)
    n_single_term = size(MM, 1)
    matrices = []
    
    ver = zeros(Int, size(MM[1,:]))
    matri = reshape(ver, 1, length(ver))
    push!(matrices, matri)

    split_num = [1] 
    
    for n in 0:x_max_term
        row_combinations = collect(combinations(1:n_single_term, n))
        
        push!(split_num, length(row_combinations))
        push!(split_num, length(row_combinations) + 1)

        if n != 0
            for selected_indices in row_combinations
                new_matrix = MM[selected_indices, :]
                push!(matrices, new_matrix)
            end
        end
    end

    for i in 1:Int((length(split_num)-1)/2 - 1)
        split_num[2+2*i] = split_num[2+2*i] + split_num[2*i]
        split_num[3+2*i] = split_num[2+2*i] + 1
    end

    return matrices, split_num
end

"""
Fit the rational multivariate polynomial coefficients using Weighted Least Squares (WLS).
This acts as Stage 1 to generate high-quality initial guesses.
"""
function fit_weighted(MM_numerator, MM_denominator, x_known, y_known, weights)
    n_train = size(x_known, 1)
    w = sqrt.(weights)
    matrix_1 = ones(n_train, 1)

    XM_train_numerator = ones(n_train, size(MM_numerator, 1))
    XM_train_denominator = ones(n_train, size(MM_denominator, 1))

    for i in axes(MM_numerator, 1)        
        XM_train_numerator[:, i] = prod(x_known .^ MM_numerator[i, :]', dims=2)        
    end

    for i in axes(MM_denominator, 1)   
        XM_train_denominator[:, i] = prod(x_known .^ MM_denominator[i, :]', dims=2)        
    end
    
    if all(x -> x == 0, MM_numerator)
        XM_train = hcat(matrix_1, -XM_train_denominator .* y_known)
    elseif all(x -> x == 0, MM_denominator)
        XM_train = hcat(matrix_1, XM_train_numerator)
    else
        XM_train = hcat(matrix_1, XM_train_numerator, -XM_train_denominator .* y_known)
    end
    
    lhs = XM_train .* w
    rhs = y_known .* w
    coeff = lhs \ rhs
    
    return coeff
end

"""
Calculate predicted values based on the fitted coefficients.
"""
function evaluate_poly(coeff, MM_1, MM_2, x_known)
    n_train = size(x_known, 1)
    XM_train_numerator = ones(n_train, size(MM_1, 1))
    XM_train_denominator = ones(n_train, size(MM_2, 1))

    if all(x -> x == 0, MM_1)
        for i in axes(MM_2, 1)        
            XM_train_denominator[:, i] = prod(x_known .^ MM_2[i, :]', dims=2)        
        end
        # 强制将分子变成一个一维向量
        numerator = fill(coeff[1], n_train) 
        # 使用 vec() 确保矩阵乘法后转为一维向量
        denominator = 1.0 .+ vec(XM_train_denominator * coeff[2:end])
        
    elseif all(x -> x == 0, MM_2)
        matrix_2 = ones(n_train, 1) 
        for i in axes(MM_1, 1)        
            XM_train_numerator[:, i] = prod(x_known .^ MM_1[i, :]', dims=2)        
        end
        XM_train_numerator = hcat(matrix_2, XM_train_numerator)
        numerator = vec(XM_train_numerator * coeff)
        denominator = ones(n_train) 
        
    else
        for i in axes(MM_1, 1) 
            XM_train_numerator[:, i] = prod(x_known .^ MM_1[i, :]', dims=2)        
        end
        matrix_2 = ones(n_train, 1)
        XM_train_numerator = hcat(matrix_2, XM_train_numerator)
        
        for i in axes(MM_2, 1)        
            XM_train_denominator[:, i] = prod(x_known .^ MM_2[i, :]', dims=2)        
        end
        
        nn = 1 + size(MM_1, 1) 
        numerator = vec(XM_train_numerator * coeff[1:nn])
        denominator = 1.0 .+ vec(XM_train_denominator * coeff[nn+1:end])
    end

    # 最终返回前，再次确保除法结果是标准的一维 Vector
    return vec(numerator ./ denominator)
end

"""
Evaluate model performance metrics: RMSE, Relative RMSE, and R-Squared.
"""
function performance_evaluation(ypred, y_known)
    testMSE = mean((y_known .- ypred).^2)
    rmse = sqrt(testMSE)
    rrmse = rmse / maximum(abs.(y_known))
    RSquared = 1 - sum((ypred .- y_known).^2) / sum((y_known .- mean(y_known)).^2)
    return rmse, rrmse, RSquared
end 

"""
Find the smallest RMSE pairs and their indices in the given shared matrix array.
"""
function find_smallest_pairs(ssr_matrix, split_num_numerator, split_num_denominator, n::Int=1)
    pairs = Tuple{Float64, Int, Int}[]
    
    for i in axes(ssr_matrix, 1)
        for j in axes(ssr_matrix, 2)
            val = ssr_matrix[i, j]
            if length(pairs) < n
                push!(pairs, (val, i + split_num_numerator - 1, j + split_num_denominator - 1))
            else
                max_val, _, _ = maximum(pairs)
                if val < max_val
                    pairs[argmax(map(p -> p[1], pairs))] = (val, i + split_num_numerator - 1, j + split_num_denominator - 1)
                end
            end
        end
    end
    sort!(pairs, by=p -> p[1])
    return pairs[1:min(n, length(pairs))]
end

"""
Generate the string representation of the mathematical function.
"""
function get_the_function_expression(coeff, MM_numerator, MM_denominator)
    coeff_length = length(coeff)
    str_function_expression = "y = (" * @sprintf("%f", coeff[1])

    if !all(x -> x == 0, MM_numerator)
        for i in axes(MM_numerator, 1)
            term_power = MM_numerator[i,:]
            sign_str = coeff[i+1] >= 0 ? " +" : " "
            str_function_expression *= sign_str * @sprintf("%f", coeff[i+1]) 

            for j in eachindex(term_power)
                if term_power[j] == 1
                    str_function_expression *= " * x$(j)"
                elseif term_power[j] > 1
                    str_function_expression *= " * x$(j)^$(term_power[j])"
                end
            end
        end
    end
    str_function_expression *= ")"

    if !all(x -> x == 0, MM_denominator)
        str_function_expression *= " / (1.0"
        for i in axes(MM_denominator, 1)
            term_power = MM_denominator[i,:]
            coeff_idx = coeff_length - size(MM_denominator, 1) + i
            sign_str = coeff[coeff_idx] >= 0 ? " +" : " "
            str_function_expression *= sign_str * @sprintf("%f", coeff[coeff_idx]) 
            
            for j in eachindex(term_power)
                if term_power[j] == 1
                    str_function_expression *= " * x$(j)"
                elseif term_power[j] > 1
                    str_function_expression *= " * x$(j)^$(term_power[j])"
                end
            end
        end
        str_function_expression *= ")" 
    end

    return str_function_expression
end

# ==============================================================================
# Main API execution function (Two-Stage Rational Learning)
# ==============================================================================

"""
    run_rational_learning(X::AbstractMatrix, y::AbstractVector; kwargs...)

Main entry point to perform the rational multivariate polynomial learning.
Uses Weighted Least Squares (WLS) for structural grid search, followed by 
Levenberg-Marquardt (LM) nonlinear optimization for coefficient refinement.

# Arguments
- `X`: Feature matrix
- `y`: Target vector
- `degree`: Maximum power sum of a single term (default: 1)
- `dim`: Number of unknown variables (default: 2)
- `max_num_terms`: Maximum number of numerator terms (default: 2)
- `max_den_terms`: Maximum number of denominator terms (default: 2)
- `use_lm_optimization`: Enable Levenberg-Marquardt fine-tuning (default: true)
- `file_out_path`: Path to save the detailed text log
- `csv_out_path`: Path to save the structured CSV results
"""
function run_rational_learning(
    X::AbstractMatrix, 
    y::AbstractVector; 
    degree::Int = 1, 
    dim::Int = 2, 
    max_num_terms::Int = 2, 
    max_den_terms::Int = 2,
    use_lm_optimization::Bool = true,
    file_out_path::String = "LM_results.txt",
    csv_out_path::String = "LM_results.csv"
)
    num_train = size(y, 1)
    weights = ones(num_train)

    # Initialize combinations
    MM = get_all_single_term_power_combination(degree, dim)
    array_num_terms = get_the_array(max_num_terms, max_den_terms)

    MMs_numerator, split_num_numerator = get_power_combination(MM, max_num_terms)
    num_MMs_numerator = size(MMs_numerator, 1)

    MMs_denominator, split_num_denominator = get_power_combination(MM, max_den_terms)
    num_MMs_denominator = size(MMs_denominator, 1)

    println("Total Numerator Combinations: ", num_MMs_numerator)
    println("Total Denominator Combinations: ", num_MMs_denominator)
    println("Total Structural Evaluations: ", num_MMs_numerator * num_MMs_denominator)
    println("Learning Model, please wait...")

    ssr_matrix = zeros(Float64, num_MMs_numerator, num_MMs_denominator)

    # Initialize output files
    open(file_out_path, "w") do io
        write(io, "Max single term power: $degree\n")
        write(io, "Number of variables: $dim\n")
        write(io, "Max numerator terms: $max_num_terms\n")
        write(io, "Max denominator terms: $max_den_terms\n")
        write(io, "LM Optimization Enabled: $use_lm_optimization\n\n")
    end

    csv_header = ["num_numerator_terms", "num_denominator_terms", "rmse", "r_squared", "rrmse", "function_expression", "coefficients", "numerator_powers", "denominator_powers"]
    open(csv_out_path, "w") do io
        write(io, join(csv_header, ",") * "\n")
    end

    # Process each structural combination
    for combination in array_num_terms
        row_start = split_num_numerator[2 * combination[1] + 1]
        row_end = split_num_numerator[2 * combination[1] + 2]
        col_start = split_num_denominator[2 * combination[2] + 1]
        col_end = split_num_denominator[2 * combination[2] + 2]

        # Phase 1: Grid Search with fast WLS
        Threads.@threads for i in row_start:row_end
            for j in col_start:col_end
                coeff = fit_weighted(MMs_numerator[i], MMs_denominator[j], X, y, weights)
                y_pred = evaluate_poly(coeff, MMs_numerator[i], MMs_denominator[j], X)
                rmse, _, _ = performance_evaluation(y_pred, y)
                ssr_matrix[i, j] = rmse 
            end
        end

        calc_field = ssr_matrix[row_start:row_end, col_start:col_end]
        best_pairs = find_smallest_pairs(calc_field, row_start, col_start, 1)
        
        if isempty(best_pairs) continue end
        
        best_rmse, min_num_idx, min_den_idx = best_pairs[1]

        # --- Two-Stage Coefficient Optimization ---
        
        # Stage 1: Generate Initial Guess (p0) using WLS
        best_coeff_wls = fit_weighted(MMs_numerator[min_num_idx], MMs_denominator[min_den_idx], X, y, weights)
        best_coeff = best_coeff_wls
        
        # Stage 2: Levenberg-Marquardt (LM) Fine-tuning
        if use_lm_optimization
            # 构造符合 LsqFit.curve_fit 接口的匿名函数模型
            model_for_lm(x_data, p) = evaluate_poly(p, MMs_numerator[min_num_idx], MMs_denominator[min_den_idx], x_data)
            
            try
                fit_result = curve_fit(model_for_lm, X, y, best_coeff_wls)
                best_coeff = fit_result.param
            catch e
                # 容错：如果发生奇异雅可比矩阵或不收敛等问题，回退到 WLS 的系数
                println("  [Warning] LM Optimization failed for terms $(combination). Falling back to WLS. Error: $e")
                best_coeff = best_coeff_wls
            end
        end

        # ------------------------------------------

        best_y_pred = evaluate_poly(best_coeff, MMs_numerator[min_num_idx], MMs_denominator[min_den_idx], X)
        rmse, rrmse, RSquared = performance_evaluation(best_y_pred, y)
        func_expr = get_the_function_expression(best_coeff, MMs_numerator[min_num_idx], MMs_denominator[min_den_idx])

        println("Current Numerator/Denominator Terms (Excl. Constant): ", combination)
        println("Optimal R-Squared: $RSquared \n")

        # Write txt results
        open(file_out_path, "a") do io
            write(io, "\nComputed Terms (Num, Den): $(combination)\n")
            write(io, "Symbolic Expression: $func_expr\n")
            write(io, "Coefficients: $best_coeff\n")
            write(io, "RMSE: $rmse\n")
            write(io, "Relative RMSE: $rrmse\n")
            write(io, "R-Squared: $RSquared\n\n")
        end

        # Write csv results
        csv_row = [
            combination[1], combination[2], rmse, RSquared, rrmse,
            "\"$func_expr\"", "\"$best_coeff\"", 
            "\"$(MMs_numerator[min_num_idx])\"", "\"$(MMs_denominator[min_den_idx])\""
        ]
        open(csv_out_path, "a") do io
            write(io, join(csv_row, ",") * "\n")
        end
    end

    println("Calculation complete. Check $file_out_path and $csv_out_path.")
end

end # module