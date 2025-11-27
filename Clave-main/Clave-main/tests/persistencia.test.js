jest.mock('../src/persistencia/database'); // Mocka o módulo Database

const Database = require('../src/persistencia/database');
const AlunoRepository = require('../src/persistencia/AlunoRepository');
const ProfessorRepository = require('../src/persistencia/ProfessorRepository');
const AvaliacaoRepository = require('../src/persistencia/AvaliacaoRepository');
const AulaRepository = require('../src/persistencia/AulaRepository');
const StatusAulaManager = require('../src/persistencia/StatusAulaManager');
const Aluno = require('../src/dominio/Aluno');
const Professor = require('../src/dominio/Professor');
const Avaliacao = require('../src/dominio/Avaliacao');
const StatusAula = require('../src/dominio/StatusAula');

describe('Camada de Persistência', () => {

  let mockPool;

  beforeEach(() => {
    mockPool = { query: jest.fn() };
    Database.getPool.mockReturnValue(mockPool);
  });

  // --- ALUNO REPOSITORY ---
  test('AlunoRepository.salvar deve inserir aluno e retornar o ID', async () => {
    const repo = new AlunoRepository();
    const aluno = new Aluno(1, 'Ana', 'ana@email.com');

    mockPool.query.mockResolvedValue({ rows: [{ id: 10 }] });

    const id = await repo.salvar(aluno);
    expect(mockPool.query).toHaveBeenCalledWith(
      'INSERT INTO aluno (nome, email) VALUES ($1, $2) RETURNING id',
      ['Ana', 'ana@email.com']
    );
    expect(id).toBe(10);
  });

  test('AlunoRepository.buscarPorId deve retornar um objeto Aluno', async () => {
    const repo = new AlunoRepository();

    mockPool.query.mockResolvedValue({
      rows: [{ id: 1, nome: 'Ana', email: 'ana@email.com' }]
    });

    const aluno = await repo.buscarPorId(1);
    expect(aluno).toBeInstanceOf(Aluno);
    expect(aluno.obterNome()).toBe('Ana');
  });

  // --- PROFESSOR REPOSITORY ---
  test('ProfessorRepository.salvar deve salvar professor e retornar o ID', async () => {
    const repo = new ProfessorRepository();
    const professor = {
      obterNome: () => 'Carlos',
      obterEmail: () => 'carlos@email.com',
      obterInstrumento: () => 'Violão',
      obterPrecoHora: () => 100
    };

    mockPool.query.mockResolvedValue({ rows: [{ id: 22 }] });

    const id = await repo.salvar(professor);
    expect(id).toBe(22);
    expect(mockPool.query).toHaveBeenCalled();
  });

  // --- AVALIAÇÃO REPOSITORY ---
  test('AvaliacaoRepository.salvar deve inserir avaliação e retornar o ID', async () => {
    const repo = new AvaliacaoRepository();
    const avaliacao = {
      obterAulaId: () => 1,
      obterNota: () => 5,
      obterComentario: () => 'Excelente aula!'
    };

    mockPool.query.mockResolvedValue({ rows: [{ id: 33 }] });

    const id = await repo.salvar(avaliacao);
    expect(id).toBe(33);
    expect(mockPool.query).toHaveBeenCalledWith(
      expect.stringContaining('INSERT INTO avaliacao'),
      [1, 5, 'Excelente aula!']
    );
  });

  // --- STATUS AULA MANAGER ---
  test('StatusAulaManager deve validar status e formatar corretamente', () => {
    expect(StatusAulaManager.isValidStatus(StatusAula.CONFIRMADA)).toBe(true);
    expect(StatusAulaManager.isValidStatus('INVALIDO')).toBe(false);

    expect(StatusAulaManager.formatarStatus('CANCELADA_PELO_ALUNO'))
      .toBe('Cancelada pelo Aluno');
  });

  test('StatusAulaManager deve identificar transições válidas de status', () => {
    expect(StatusAulaManager.isTransicaoValida('SOLICITADA', 'CONFIRMADA')).toBe(true);
    expect(StatusAulaManager.isTransicaoValida('REALIZADA', 'CONFIRMADA')).toBe(false);
  });

  // --- BUSCAR AULAS POR STATUS ---
  test('StatusAulaManager.buscarAulasPorStatus deve lançar erro se status for inválido', async () => {
    await expect(StatusAulaManager.buscarAulasPorStatus('ERRADO'))
      .rejects.toThrow('Status inválido');
  });
});
