const Aluno = require('./src/dominio/Aluno');
const Professor = require('./src/dominio/Professor');
const AlunoRepositorio = require('./src/persistencia/AlunoRepository');
const ProfessorRepositorio = require('./src/persistencia/ProfessorRepository');
const Database = require('./src/persistencia/database');

// A função principal precisa ser assíncrona para usar 'await'
async function main() {
    console.log("--- INICIANDO TESTES DA CAMADA DE PERSISTÊNCIA ---");

    try {
        // Instanciando os repositórios
        const alunoRepo = new AlunoRepositorio();
        const professorRepo = new ProfessorRepositorio();

        // 1. TESTE DE CRIAÇÃO (CREATE)
        console.log("\n[TESTE] Criando um novo aluno...");
        const novoAluno = new Aluno(null, 'Luana Fernandes', 'luana.fernandes@gmail.com'); // ID é null pois será gerado pelo DB
        const alunoId = await alunoRepo.salvar(novoAluno);
        console.log(`> Aluno "Luana Fernandes" criado com sucesso! ID: ${alunoId}`);

        // 2. TESTE DE LEITURA (READ)
        console.log("\n[TESTE] Buscando o aluno recém-criado pelo ID...");
        const alunoEncontrado = await alunoRepo.buscarPorId(alunoId);
        if (alunoEncontrado) {
            console.log(`> Encontrado: ${alunoEncontrado.obterNome()} (Email: ${alunoEncontrado.obterEmail()})`);
        } else {
            console.error("> ERRO: Aluno não encontrado!");
        }

        // 3. TESTE DE ATUALIZAÇÃO (UPDATE)
        console.log("\n[TESTE] Atualizando o nome do aluno...");
        alunoEncontrado.alterarNome("Luana Fernandes da Silva"); // Supondo que você tenha um setter no domínio
        await alunoRepo.atualizar(alunoEncontrado);
        const alunoAtualizado = await alunoRepo.buscarPorId(alunoId);
        console.log(`> Nome atualizado para: ${alunoAtualizado.obterNome()}`);


        // 4. TESTE DE LEITURA DE TODOS (READ ALL)
        console.log("\n[TESTE] Buscando todos os professores...");
        const todosProfessores = await professorRepo.buscarTodos();
        console.log(`> Total de professores no banco: ${todosProfessores.length}`);
        todosProfessores.forEach(prof => console.log(`  - ${prof.obterNome()}`));

        
        // 5. TESTE DE EXCLUSÃO (DELETE)
        console.log("\n[TESTE] Deletando o aluno de teste...");
        await alunoRepo.deletar(alunoId);
        const alunoDeletado = await alunoRepo.buscarPorId(alunoId);
        if (!alunoDeletado) {
            console.log("> Aluno deletado com sucesso!");
        } else {
            console.error("> ERRO: O aluno ainda existe no banco.");
        }

    } catch (error) {
        console.error("\n*** UM ERRO OCORREU DURANTE OS TESTES ***");
        console.error(error);
    } finally {
    // Chama o método correto para fechar o pool de conexões
    await Database.closePool();
}
}

// Executa a função principal
main();